# WFR-Gossip vs Gossipsub Simulator

This repo contains a discrete-event simulation that compares Ethereum’s
standard Gossipsub protocol with an alternative latency-aware variant
(WFR-Gossip). It sweeps the **D_ROBUST** parameter, records coverage,
latency, hops, bandwidth, and duplicates, and outputs publication-grade
figures (PNG + PDF).

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python simulator.py          # runs the sweep and plots the figures
