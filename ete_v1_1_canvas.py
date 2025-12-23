#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
═══════════════════════════════════════════════════════════════════════════════
ETE v1.1 FINAL - COGNITIVE EFFICIENCY SIMULATOR
Endobiotic Totality Engine
═══════════════════════════════════════════════════════════════════════════════

PURPOSE
- Compare two cognitive strategies under identical task environments
- Demonstrate why high-gain / high-rigidity strategies collapse long-term
- Provide an EXECUTABLE gate, not an optimizer

IMPORTANT
- This is a computational thought experiment
- NOT a biological model
- NOT a universal law
- All quantities are abstract cost / efficiency proxies
"""

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT SAFETY (Windows / Python 3.12)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

# Optional: uncomment for deterministic debugging (default OFF by design)
# np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CONSTANTS (Validated — DO NOT MODIFY casually)
# ─────────────────────────────────────────────────────────────────────────────

ITERATIONS = 10_000
SWITCH_TASK_PROBABILITY = 0.05
NOISE_LEVEL = 2.0

PERFORMANCE_CAP = 100.0
MIN_FLEXIBILITY = 0.05

CRASH_DECAY = 0.999
RIGIDITY_GROWTH = 0.0005
CRASH_ENERGY_MULTIPLIER = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# COGNITIVE AGENT (Physics Layer)
# ─────────────────────────────────────────────────────────────────────────────

class CognitiveAgent:
    """
    Abstract cognitive processor.

    State variables:
      G (neural_gain): signal amplification
      R (rigidity): adaptability vs perseveration
      Energy: cumulative cost
    """

    def __init__(self, name: str, baseline_gain: float,
                 rigidity: float, crash_point: int | None):
        self.name = name
        self.neural_gain = float(baseline_gain)
        self.rigidity = float(rigidity)
        self.crash_point = crash_point

        self.energy_consumed: float = 0.0
        self.performance_history: list[float] = []
        self.state_history: list[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Core processing step
    # ─────────────────────────────────────────────────────────────────────────

    def process_task(self, signal: float, is_switch: bool, t: int) -> float:
        """
        Execute one timestep of cognitive processing.
        """

        crashed = self._check_and_apply_crash(t)

        # Signal amplification
        processed = signal * self.neural_gain

        # Flexibility vs perseveration
        if is_switch:
            flexibility = max(MIN_FLEXIBILITY, 1.0 - self.rigidity)
            performance = processed * flexibility
        else:
            performance = processed * (1.0 + self.rigidity * 0.1)

        # Saturation (biological / control limit proxy)
        performance = np.clip(performance, 0.0, PERFORMANCE_CAP)

        # Gain-dependent noise
        noise_scale = self._noise_scale(crashed)
        performance += np.random.normal(0.0, noise_scale)

        # Prevent pathological negative output
        performance = max(0.0, performance)

        # Energy accounting
        self._accumulate_energy(crashed)

        self.performance_history.append(performance)
        return performance

    # ─────────────────────────────────────────────────────────────────────────
    # Internal mechanics
    # ─────────────────────────────────────────────────────────────────────────

    def _check_and_apply_crash(self, t: int) -> bool:
        if self.crash_point is not None and t > self.crash_point:
            self.neural_gain *= CRASH_DECAY
            self.rigidity = min(1.0, self.rigidity + RIGIDITY_GROWTH)
            self.state_history.append("Crash")
            return True

        self.state_history.append(
            "High" if self.neural_gain > 1.5 else "Normal"
        )
        return False

    def _noise_scale(self, crashed: bool) -> float:
        if crashed:
            return NOISE_LEVEL * (1.0 + self.rigidity) * self.neural_gain
        return NOISE_LEVEL

    def _accumulate_energy(self, crashed: bool) -> None:
        multiplier = CRASH_ENERGY_MULTIPLIER if crashed else 1.0
        cost = (self.neural_gain * (1.0 + self.rigidity)) * multiplier
        self.energy_consumed += cost


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation() -> None:
    # Agents
    meth = CognitiveAgent(
        name="Meth-State",
        baseline_gain=2.5,
        rigidity=0.95,
        crash_point=6000
    )

    will = CognitiveAgent(
        name="Will-State",
        baseline_gain=1.618,
        rigidity=0.1,
        crash_point=None
    )

    # Task stream
    signals = np.random.uniform(30, 50, ITERATIONS)
    switches = np.random.rand(ITERATIONS) < SWITCH_TASK_PROBABILITY

    # Main loop
    for t in range(ITERATIONS):
        meth.process_task(signals[t], switches[t], t)
        will.process_task(signals[t], switches[t], t)

    # Metrics
    meth_perf = np.array(meth.performance_history)
    will_perf = np.array(will.performance_history)

    meth_psi = meth_perf.sum() / (meth.energy_consumed + 1.0)
    will_psi = will_perf.sum() / (will.energy_consumed + 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────────────────────────────────

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.plot(meth_perf, alpha=0.6, label="Meth-State", color="red")
    plt.plot(will_perf, alpha=0.8, label="Will-State", color="green")
    plt.axvline(6000, linestyle="--", color="black")
    plt.title("Performance Over Time")
    plt.xlabel("Time")
    plt.ylabel("Performance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(meth_perf), label="Meth-State", color="red")
    plt.plot(np.cumsum(will_perf), label="Will-State", color="green")
    plt.axvline(6000, linestyle="--", color="black")
    plt.title("Cumulative Output (Efficiency Proxy)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Performance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # Console Report
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 68)
    print("ETE v1.1 RESULT SUMMARY")
    print("=" * 68)
    print(f"Meth Ψ  : {meth_psi:.4f}")
    print(f"Will Ψ  : {will_psi:.4f}")
    print(f"Ratio   : {will_psi / meth_psi:.2f}x efficiency advantage")
    print("=" * 68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_simulation()
