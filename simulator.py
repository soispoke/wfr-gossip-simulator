#!/usr/bin/env python3
"""
Gossipsub vs. WFR-Gossip simulator
• Parameter sweep: D_ROBUST = 1 … 8
• Records: coverage, P90 / mean / std latency, hops, bandwidth, duplicates
• Produces publication-grade figures (PNG + PDF)

Author: soispoke
"""
import os, random, simpy, networkx as nx, numpy as np, pandas as pd
import matplotlib.pyplot as plt; from matplotlib import cm; import matplotlib.ticker as mticker

# ─────────── Matplotlib (journal style) ────────────
plt.rcParams.update({
    "figure.dpi":300, "savefig.dpi":300, "figure.autolayout":True,
    "font.family":"serif", "font.serif":["Times New Roman","Times","DejaVu Serif"],
    "font.size":11, "axes.labelsize":12, "axes.titlesize":13,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.alpha":0.4,
    "xtick.labelsize":10, "ytick.labelsize":10,
    "legend.fontsize":10, "lines.markersize":5,
})
CB = cm.get_cmap("tab10").colors             # colour palette

# ───────────────────── Config ─────────────────────
class Config:
    NUM_NODES        = 10_000
    AVG_CONNECTIVITY = 50
    MESH_DEGREE      = 8
    MESSAGE_SIZE_KB  = 90
    SIM_DURATION     = 30           # seconds (sim-time)

    LAT_BASE   = 0.010
    LAT_JITTER = 0.005
    LAT_GEO_F  = 0.150

    RESULTS_FILE = "simulation_results.csv"
    FIGURE_FILE_1 = "wfr_gossip_perf_side_by_side.png"
    FIGURE_FILE_2 = "wfr_gossip_bandwidth_breakdown.png"

random.seed(42); np.random.seed(42)

# ─────────────────── Node ───────────────────
class Node:
    def __init__(self, env, nid, proto, sim, d_r):
        self.env, self.id, self.proto, self.sim, self.d_r = env, nid, proto, sim, d_r
        self.peers, self.mesh, self.seen = [], [], set()

    def connect(self, neigh):
        self.peers = list(neigh)
        if self.proto == "gossipsub" and self.peers:
            self.mesh = random.sample(self.peers,
                                      min(len(self.peers), Config.MESH_DEGREE))

    def receive(self, msg, sender, rtt_in):
        mid, hops = msg
        if mid in self.seen:
            self.sim.m["duplicates"] += 1
            return
        self.seen.add(mid)
        self.sim.m["arrivals"][self.id] = self.env.now
        self.sim.m["hops"][self.id] = hops
        yield self.env.timeout(0.001 + random.uniform(0, 0.002))
        nxt = (mid, hops + 1)
        if self.proto == "gossipsub":
            self._fwd_mesh(nxt, sender)
        else:
            self._fwd_wfr(nxt, sender, rtt_in)

    # --- Gossipsub: fixed mesh of 8 peers
    def _fwd_mesh(self, msg, sender):
        for p in self.mesh:
            if p != sender:
                self.sim.send(self.id, p, msg)

    # --- WFR-Gossip: D_ROBUST random peers + downhill optimisation
    def _fwd_wfr(self, msg, sender, rtt_in):
        if not self.peers:
            return
        # robustness
        chosen = set(random.sample([p for p in self.peers if p != sender],
                                   min(self.d_r, len(self.peers) - 1)))
        # efficiency
        slots = Config.MESH_DEGREE - len(chosen)
        if slots:
            best = sorted(self.peers,
                          key=lambda p: self.sim.lat[(self.id, p)])[:Config.MESH_DEGREE]
            for p in best:
                if slots == 0:
                    break
                if p == sender or p in chosen:
                    continue
                if self.sim.lat[(self.id, p)] < rtt_in:
                    chosen.add(p); slots -= 1
        # send
        for p in chosen:
            self.sim.send(self.id, p, msg)

# ─────────────────── Simulator ───────────────────
class Simulator:
    def __init__(self, g, lat): self.g, self.lat = g, lat
    def run(self, proto, d=0):
        self.env = simpy.Environment()
        self.m = {"arrivals": {}, "hops": {}, "duplicates": 0, "sent": 0}
        self.nodes = {n: Node(self.env, n, proto, self, d) for n in self.g.nodes()}
        for n in self.nodes: self.nodes[n].connect(self.g.neighbors(n))
        origin = random.choice(list(self.nodes))
        self.env.process(self.nodes[origin].receive(("msg", 0), origin, 0))
        self.env.run(until=Config.SIM_DURATION)
        return self._stats(proto, d)

    def send(self, src, dst, msg):
        self.m["sent"] += 1; rtt = self.lat[(src, dst)]
        def deliver():
            yield self.env.timeout(rtt)
            self.env.process(self.nodes[dst].receive(msg, src, rtt))
        self.env.process(deliver())

    def _stats(self, proto, d):
        N = len(self.g)
        times = np.array(list(self.m["arrivals"].values()))
        hops  = list(self.m["hops"].values())
        return {
            "Protocol" : "Gossipsub" if proto == "gossipsub" else "WFR-Gossip",
            "D_ROBUST" : d if proto != "gossipsub" else "N/A",
            "Coverage (%)"     : round(100 * len(times) / N, 2),
            "P90 Time (ms)"    : round(np.percentile(times, 90) * 1000, 2) if len(times) else float("inf"),
            "Mean Time (ms)"   : round(times.mean()*1000, 2) if len(times) else float("inf"),
            "Std Time (ms)"    : round(times.std(ddof=0)*1000, 2) if len(times) else float("inf"),
            "Mean Hops"        : round(np.mean(hops), 2) if hops else 0,
            "P90 Hops"         : round(np.percentile(hops, 90), 2) if hops else 0,
            "Mean Duplicates"  : round(self.m["duplicates"] / N, 2),
            "Total Egress (MB)": round(self.m["sent"] * Config.MESSAGE_SIZE_KB / 1024, 2),
            "Wasted Bandwidth (MB)": round(self.m["duplicates"] * Config.MESSAGE_SIZE_KB / 1024, 2),
        }

# ─────────────────── Helpers ───────────────────
def build_network():
    g = nx.barabasi_albert_graph(Config.NUM_NODES, Config.AVG_CONNECTIVITY // 2)
    pos = {n: np.random.rand(2) for n in g.nodes()}
    lat = {}
    for i in g.nodes():
        for j in g.neighbors(i):
            d = np.linalg.norm(pos[i] - pos[j])
            r = Config.LAT_BASE + d * Config.LAT_GEO_F + random.uniform(0, Config.LAT_JITTER)
            lat[(i, j)] = lat[(j, i)] = r
    return g, lat

def run_experiment(g, lat):
    sim = Simulator(g, lat)
    rows = [sim.run("gossipsub")]
    for d in range(1, Config.MESH_DEGREE + 1):
        rows.append(sim.run("WFR-Gossip", d))
    return pd.DataFrame(rows)

def save_fig(fig, path):
    fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    fig.savefig(path, bbox_inches="tight")

# ─────────────────── Plotting ───────────────────
def plot_results(df):
    if df.empty:
        print("No results."); return
    d_num = pd.to_numeric(df["D_ROBUST"], errors="coerce")
    wfr   = df[(df["Protocol"] == "WFR-Gossip") & (d_num >= 1)].copy()
    wfr["D_ROBUST"] = d_num[(df["Protocol"] == "WFR-Gossip") & (d_num >= 1)]
    wfr = wfr.set_index("D_ROBUST")
    gs  = df[df["Protocol"] == "Gossipsub"].iloc[0]

    # ── Figure 1: side-by-side
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.3*2, 4.9))

    # (a) Egress vs coverage
    axL2 = axL.twinx()
    axL.plot(wfr.index, wfr["Total Egress (MB)"], "o-", color=CB[0], label="Total egress")
    axL.axhline(gs["Total Egress (MB)"], color=CB[0], ls="--")
    axL.set_xlabel("D_ROBUST"); axL.set_ylabel("MiB", color=CB[0])
    axL2.plot(wfr.index, wfr["Coverage (%)"], "s-", color=CB[1], label="Coverage")
    axL2.axhline(gs["Coverage (%)"], color=CB[1], ls="--")
    axL2.set_ylabel("Coverage (%)", color=CB[1])
    axL2.yaxis.set_major_formatter(mticker.PercentFormatter())
    hL1,lL1=axL.get_legend_handles_labels(); hL2,lL2=axL2.get_legend_handles_labels()
    axL.legend(hL1+hL2, lL1+lL2, frameon=False, loc="lower right")
    axL.set_title("(a) Egress vs coverage")

    # (b) Latency plot with dual y-axis
    axR.plot(wfr.index, wfr["P90 Time (ms)"], "^-", color=CB[2], label="P90 latency")
    axR.axhline(gs["P90 Time (ms)"], color=CB[2], ls="--")
    axR.set_xlabel("D_ROBUST"); axR.set_ylabel("P90 latency (ms)", color=CB[2])
    axR2 = axR.twinx()
    axR2.plot(wfr.index, wfr["Mean Time (ms)"], "o--", color=CB[5], label="Mean latency")
    axR2.axhline(gs["Mean Time (ms)"], color=CB[5], ls="--")
    axR2.set_ylabel("Mean latency (ms)", color=CB[5])
    hR1,lR1=axR.get_legend_handles_labels(); hR2,lR2=axR2.get_legend_handles_labels()
    axR.legend(hR1+hR2, lR1+lR2, frameon=False, loc="upper right")
    axR.set_title("(b) Propagation latency")
    save_fig(fig, Config.FIGURE_FILE_1); plt.close(fig)

    # ── Figure 2: bandwidth breakdown
    fig2, ax = plt.subplots(figsize=(6.3, 4.8))
    ax.plot(wfr.index, wfr["Total Egress (MB)"], "o-", color=CB[3], label="Total egress")
    ax.plot(wfr.index, wfr["Wasted Bandwidth (MB)"], "s-", color=CB[4], label="Wasted egress")
    ax.axhline(gs["Total Egress (MB)"], color=CB[3], ls="--")
    gs_waste = gs["Total Egress (MB)"] - (gs["Coverage (%)"]/100 *
                Config.NUM_NODES * Config.MESSAGE_SIZE_KB / 1024)
    ax.axhline(gs_waste, color=CB[4], ls="--")
    ax.set_xlabel("D_ROBUST"); ax.set_ylabel("MiB")
    ax.set_title("Bandwidth breakdown")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig2, Config.FIGURE_FILE_2); plt.close(fig2)
    print("Figures saved →", Config.FIGURE_FILE_1, "&", Config.FIGURE_FILE_2)

# ─────────────────── Main ───────────────────
if __name__ == "__main__":
    if os.path.exists(Config.RESULTS_FILE):
        df = pd.read_csv(Config.RESULTS_FILE)
        wanted = set(map(float, range(1, 9)))
        got = set(pd.to_numeric(df["D_ROBUST"], errors="coerce").dropna())
        if wanted - got:
            os.remove(Config.RESULTS_FILE); df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        g, lat = build_network()
        df = run_experiment(g, lat)
        df.to_csv(Config.RESULTS_FILE, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:8.2f}")
    print(df.to_string(index=False))
    plot_results(df)
