# ETE v1.1 — Executable Thought Experiment

> **An executable thought experiment that simulates the long-term efficiency and collapse dynamics of different decision strategies.**

---

## Why

Many strategies that maximize short-term output appear effective at first,
yet fail catastrophically over time.

This project exists to answer a simple but often unmodeled question:

> **What is the long-term cost of high-gain, rigid strategies compared to stable, adaptive ones?**

Instead of arguing philosophically, this repository encodes the question
as a **minimal, executable simulation** that makes the trade-off visible over time.

This is **not** a model of the human brain.
It is a **strategy-level abstraction** designed to expose structural dynamics:

* output vs. cost
* gain vs. rigidity
* short-term advantage vs. long-term efficiency

---

## Model

The simulation compares two abstract decision states:

### 1. High-Gain / High-Rigidity State (`Meth-State`)

* Produces high output early
* Amplifies noise as gain increases
* Accumulates irreversible degradation
* Eventually collapses after energy depletion

### 2. Optimal-Gain / Flexible State (`Will-State`)

* Maintains moderate, stable output
* Uses controlled stochasticity
* Preserves adaptability
* Remains efficient over long horizons

The system tracks:

* instantaneous performance
* cumulative output (area under the curve)
* collapse dynamics after a critical depletion point

The model is **deterministic except for controlled noise**,
allowing reproducible experiments with interpretable variance.

---

## Assumptions

This simulation intentionally makes strong simplifications:

* Strategies are modeled as **state machines**, not agents with beliefs
* “Energy,” “rigidity,” and “noise” are abstract parameters, not biological measurements
* Collapse is modeled as an **irreversible regime change**
* No learning, recovery, or external intervention is allowed after collapse
* Parameters are chosen for **structural clarity**, not empirical fitting

These assumptions are not flaws — they are **constraints that make the dynamics legible**.

---

## Limits

What this project does **not** claim:

* ❌ It does not model neuroscience or psychology
* ❌ It does not predict individual human behavior
* ❌ It is not a medical, clinical, or behavioral theory
* ❌ It is not an AI or reinforcement learning system

This simulation should be read as:

> **a structural demonstration of strategy dynamics, not a theory of mind.**

Any interpretation beyond that is the responsibility of the reader.

---

## How to Run

### Requirements

* Python **3.10+** (tested on Python 3.12)
* `numpy`
* `matplotlib`

Install dependencies:

```bash
pip install numpy matplotlib
```

### Execute

```bash
python ete_v1_1_canvas.py
```

### Output

The script generates:

* Time-series performance comparison
* Cumulative output curves
* Visual indication of the collapse threshold

All results are produced locally with no external dependencies.

---

## Interpretation Guide (Optional Reading)

* Early dominance does **not** imply long-term superiority
* Cumulative output is a more reliable metric than peak performance
* High gain amplifies both signal **and** noise
* Rigidity accelerates collapse once adaptability is lost

If the curves surprise you, that is the point.

---

## License

MIT License — use, modify, or extend freely, with attribution.

---
