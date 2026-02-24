#!/usr/bin/env python3
"""
mycelium.py — the orchestrator. the connective tissue between organisms.

molequla has four elements:
    molequla.go   — Go
    molequla.c    — C
    molequla.js   — JavaScript
    molequla.rs   — Rust (the mouth)

mycelium connects them. it reads the field (mesh.db), computes
system-level awareness via METHOD (C-native, BLAS-accelerated),
and writes steering deltas for the mouth to consume.

usage:
    python3 mycelium.py                    # interactive REPL (default)
    python3 mycelium.py --mesh ./mesh.db   # explicit path
    python3 mycelium.py --daemon           # background daemon, no REPL
    python3 mycelium.py --interval 2.0     # step every 2 seconds
    python3 mycelium.py --once             # single step, print, exit

part of molequla. the method that connects organisms.
"""

import argparse
import asyncio
import json
import os
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path

# Add parent dir so ariannamethod is importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from ariannamethod import Method
from ariannamethod import Sentinel


# ═══════════════════════════════════════════════════════════════════════════════
# field monitor — watches organism health
# ═══════════════════════════════════════════════════════════════════════════════

class FieldMonitor:
    """tracks field health over time, detects anomalies."""

    def __init__(self):
        self.history = []       # list of steering dicts
        self.max_history = 64
        self.alerts = []

    def record(self, steering):
        self.history.append(steering)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        self._check_alerts(steering)

    def _check_alerts(self, s):
        self.alerts.clear()

        if s.get("n_organisms", 0) == 0:
            self.alerts.append("no organisms alive")
            return

        if s.get("coherence", 1.0) < 0.3:
            self.alerts.append(f"coherence low: {s['coherence']:.3f}")

        if s.get("entropy", 0) > 2.5:
            self.alerts.append(f"entropy high: {s['entropy']:.3f}")

        action = s.get("action", "")
        if len(self.history) >= 8:
            recent_actions = [h.get("action") for h in self.history[-8:]]
            if all(a == "dampen" for a in recent_actions):
                self.alerts.append("stuck in dampen loop")
            if all(a == "realign" for a in recent_actions):
                self.alerts.append("persistent incoherence")

    def status_line(self, steering):
        n = steering.get("n_organisms", 0)
        action = steering.get("action", "?")
        strength = steering.get("strength", 0)
        ent = steering.get("entropy", 0)
        syn = steering.get("syntropy", 0)
        coh = steering.get("coherence", 0)
        step = steering.get("step", 0)

        line = (f"[mycelium] step={step} organisms={n} "
                f"action={action}({strength:.2f}) "
                f"H={ent:.3f} S={syn:.3f} C={coh:.3f}")

        if self.alerts:
            line += f"  !! {'; '.join(self.alerts)}"

        return line


# ═══════════════════════════════════════════════════════════════════════════════
# drift tracker — which organisms are diverging
# ═══════════════════════════════════════════════════════════════════════════════

class DriftTracker:
    """identifies organisms drifting from the field mean."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.drifters = {}

    def update(self, method):
        self.drifters = method.field_drift()
        return self.drifters

    def report(self):
        if not self.drifters:
            return None
        lines = [f"  organism {oid}: deviation={dev:.3f}"
                 for oid, dev in sorted(self.drifters.items(), key=lambda x: -x[1])]
        return "[mycelium] drifters detected:\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# voice — how mycelium speaks about the field
# ═══════════════════════════════════════════════════════════════════════════════

class MyceliumVoice:
    """translates field state into words. not generation — awareness."""

    ACTION_SPEECH = {
        "wait":     "the field is empty. i am listening to silence.",
        "sustain":  "the field breathes. all organisms in rhythm.",
        "amplify":  "entropy falling. the field is organizing. i amplify the signal.",
        "dampen":   "entropy rising. the field dissolves. i slow the pulse.",
        "ground":   "entropy too high. i ground the field to the strongest organism.",
        "explore":  "entropy too low. the field is rigid. i open tunnels.",
        "realign":  "organisms diverge. coherence breaking. i pull them together.",
    }

    def speak(self, steering, organisms, drifters=None):
        """compose a field report in mycelium's voice."""
        n = steering.get("n_organisms", 0)
        action = steering.get("action", "wait")
        ent = steering.get("entropy", 0)
        syn = steering.get("syntropy", 0)
        coh = steering.get("coherence", 0)
        strength = steering.get("strength", 0)
        trend = steering.get("trend", 0)

        lines = []

        # opening — what am i doing
        base = self.ACTION_SPEECH.get(action, f"action: {action}.")
        if strength > 0.7:
            base = base.upper()
        lines.append(base)

        # field state
        if n == 0:
            lines.append("no organisms. the mesh is dark.")
        elif n == 1:
            lines.append(f"one organism alone. entropy {ent:.2f}.")
        else:
            # name the organisms
            names = [o.id for o in organisms[:8]]
            lines.append(f"{n} organisms: {', '.join(str(x) for x in names)}.")

            # coherence reading
            if coh > 0.8:
                lines.append(f"coherence {coh:.2f} — they think as one.")
            elif coh > 0.5:
                lines.append(f"coherence {coh:.2f} — aligned but individual.")
            elif coh > 0.3:
                lines.append(f"coherence {coh:.2f} — drifting apart.")
            else:
                lines.append(f"coherence {coh:.2f} — fragmented. they don't see each other.")

            # entropy reading
            if ent < 0.3:
                lines.append(f"entropy {ent:.2f} — crystallized. too certain.")
            elif ent < 1.0:
                lines.append(f"entropy {ent:.2f} — focused. good.")
            elif ent < 2.0:
                lines.append(f"entropy {ent:.2f} — searching.")
            else:
                lines.append(f"entropy {ent:.2f} — chaotic. losing shape.")

            # syntropy
            if syn > 0.5:
                lines.append(f"syntropy {syn:.2f} — the field has purpose.")
            elif syn > 0.2:
                lines.append(f"syntropy {syn:.2f} — some direction, not yet clear.")
            else:
                lines.append(f"syntropy {syn:.2f} — no direction. wandering.")

            # trend
            if trend > 0.1:
                lines.append("trend: organizing. entropy falling.")
            elif trend < -0.1:
                lines.append("trend: dissolving. entropy rising.")

        # drifters
        if drifters:
            for oid, dev in sorted(drifters.items(), key=lambda x: -x[1])[:3]:
                lines.append(f"  drifter: {oid} (deviation {dev:.2f})")

        return "\n".join(lines)

    def greet(self, lib_loaded, mesh_path, n_organisms):
        """startup message."""
        engine = "C+BLAS" if lib_loaded else "Python"
        lines = [
            "mycelium awakens.",
            f"METHOD engine: {engine}.",
            f"mesh: {mesh_path}.",
        ]
        if n_organisms > 0:
            lines.append(f"i see {n_organisms} organisms.")
        else:
            lines.append("the field is empty. waiting.")
        lines.append("type /field, /who, /drift, /help — or just talk.\n")
        return "\n".join(lines)




# ═══════════════════════════════════════════════════════════════════════════════
# field pulse — inspired by harmonix/haiku (ariannamethod/harmonix)
# ═══════════════════════════════════════════════════════════════════════════════

class FieldPulse:
    """
    Pulse of the field. Borrowed from harmonix/haiku PulseSnapshot.

    Three dimensions:
        novelty  — new organisms? topology changed?
        arousal  — how fast is entropy changing? (|ΔH|)
        entropy  — diversity of organism states (Shannon over individual entropies)

    The pulse modulates how strongly mycelium intervenes:
    high arousal + high novelty → be careful, field is volatile
    low arousal + low novelty → field is sleeping, maybe explore
    """

    def __init__(self):
        self.novelty = 0.0
        self.arousal = 0.0
        self.entropy = 0.0
        self._prev_organism_ids = set()
        self._prev_field_entropy = None

    def measure(self, organisms, field_entropy):
        """compute pulse from current field state."""
        # Novelty: did organisms appear or disappear?
        current_ids = set(o.id for o in organisms)
        if self._prev_organism_ids:
            new_orgs = current_ids - self._prev_organism_ids
            lost_orgs = self._prev_organism_ids - current_ids
            self.novelty = (len(new_orgs) + len(lost_orgs)) / max(1, len(current_ids))
        else:
            self.novelty = 1.0 if organisms else 0.0  # first observation = fully novel
        self._prev_organism_ids = current_ids

        # Arousal: rate of entropy change
        if self._prev_field_entropy is not None:
            self.arousal = min(1.0, abs(field_entropy - self._prev_field_entropy) * 2)
        else:
            self.arousal = 0.0
        self._prev_field_entropy = field_entropy

        # Entropy: diversity of organism states (Shannon)
        if len(organisms) >= 2:
            entropies = np.array([o.entropy for o in organisms])
            # Normalize to probabilities
            total = np.sum(entropies) + 1e-10
            probs = entropies / total
            self.entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        elif organisms:
            self.entropy = 0.0
        else:
            self.entropy = 0.0

        return self

    def as_dict(self):
        return {"novelty": round(self.novelty, 4),
                "arousal": round(self.arousal, 4),
                "entropy": round(self.entropy, 4)}


class SteeringDissonance:
    """
    Dissonance between intent and outcome. From harmonix principle:
    dissonance = mismatch between what mycelium wanted and what happened.

    If I said DAMPEN and entropy went UP → dissonance is high → increase strength.
    If I said EXPLORE and entropy went UP → consonance → maintain.

    Temperature-like: high dissonance → stronger intervention next time.
    Low dissonance → gentler touch.
    """

    # Expected effect per action (positive = entropy should go up)
    EXPECTED_EFFECT = {
        "dampen":  -1,  # entropy should decrease
        "ground":  -1,
        "amplify": +1,  # entropy should increase
        "explore": +1,
        "realign":  0,  # coherence should increase, entropy neutral
        "sustain":  0,
        "wait":     0,
    }

    def __init__(self, decay=0.9):
        self.decay = decay
        self.dissonance = 0.0  # EMA of recent dissonance
        self.history = []      # last 16 (action, expected, actual)

    def update(self, action, delta_entropy, delta_coherence):
        """
        Compute dissonance for this step.
        Returns dissonance value [0, 1].
        """
        expected_dir = self.EXPECTED_EFFECT.get(action, 0)

        if expected_dir == 0:
            # Neutral action — dissonance from absolute change
            raw = min(1.0, abs(delta_entropy) * 2)
        elif expected_dir > 0:
            # Wanted entropy up — dissonance if it went down
            raw = max(0, -delta_entropy) * 2
        else:
            # Wanted entropy down — dissonance if it went up
            raw = max(0, delta_entropy) * 2

        raw = min(1.0, raw)

        # EMA smoothing (like harmonix dissonance_ema)
        self.dissonance = self.decay * self.dissonance + (1 - self.decay) * raw

        self.history.append({"action": action, "raw": raw, "ema": self.dissonance})
        if len(self.history) > 16:
            self.history = self.history[-16:]

        return self.dissonance

    def strength_multiplier(self):
        """
        How much to scale steering strength based on dissonance.
        Borrowed from harmonix: dissonance → temperature mapping.

        Low dissonance (< 0.2) → gentle (0.5x)
        Medium (0.2-0.5) → normal (1.0x)
        High (> 0.5) → aggressive (1.5x)
        """
        if self.dissonance < 0.2:
            return 0.5 + self.dissonance * 2.5  # 0.5 → 1.0
        elif self.dissonance < 0.5:
            return 1.0                           # 1.0
        else:
            return 1.0 + (self.dissonance - 0.5) # 1.0 → 1.5


class OrganismAttention:
    """
    Organism attention map — which organisms respond to steering?
    Inspired by harmonix cloud morphing: active words get boosted.

    Organisms that respond to steering (entropy changes in expected direction)
    get higher attention weight. Unresponsive organisms decay.

    This lets mycelium focus on organisms that are "listening".
    """

    def __init__(self, boost=1.1, decay=0.99):
        self.weights = {}  # organism_id → attention weight
        self.boost = boost
        self.decay = decay

    def update(self, organisms, action, pre_entropies):
        """
        After steering, compare each organism's entropy change.
        Responsive organisms get boosted.

        pre_entropies: {org_id: entropy_before}
        """
        expected_dir = SteeringDissonance.EXPECTED_EFFECT.get(action, 0)

        for o in organisms:
            oid = o.id
            if oid not in self.weights:
                self.weights[oid] = 1.0

            if oid in pre_entropies:
                delta = o.entropy - pre_entropies[oid]
                # Check if organism responded in expected direction
                if expected_dir != 0:
                    if (expected_dir > 0 and delta > 0) or (expected_dir < 0 and delta < 0):
                        self.weights[oid] *= self.boost  # responsive
                    else:
                        self.weights[oid] *= self.decay   # unresponsive
                else:
                    self.weights[oid] *= self.decay

            # Cap weights
            self.weights[oid] = max(0.1, min(3.0, self.weights[oid]))

        # Remove dead organisms
        alive = set(o.id for o in organisms)
        self.weights = {k: v for k, v in self.weights.items() if k in alive}

    def top_organisms(self, n=3):
        """most attentive organisms."""
        return sorted(self.weights.items(), key=lambda x: -x[1])[:n]

    def report(self):
        if not self.weights:
            return "no organisms tracked."
        lines = []
        for oid, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            bar = "#" * int(w * 10)
            lines.append(f"  {oid}: {w:.3f} {bar}")
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# mycelium gamma — personality fingerprint, harmonically computed
# ═══════════════════════════════════════════════════════════════════════════════

class MyceliumGamma:
    """
    γ_myc — the mycelium's personality vector.

    not learned. not backpropped. computed from the history of decisions.
    each steering action imprints on gamma through harmonic basis functions.

    low frequencies capture tendencies (always dampening? always exploring?).
    high frequencies capture reactivity (how fast does strategy change?).

    γ_myc ∈ R^32, same dimensionality as organism gammas.
    can be compared via cosine similarity — "how aligned is mycelium with organism X?"
    """

    DIM = 32
    N_HARMONICS = 8

    def __init__(self):
        self.gamma = np.zeros(self.DIM, dtype=np.float64)
        self.magnitude = 0.0
        self._step = 0

        # fixed directions in gamma-space — one per action type
        # deterministic seed so gamma is reproducible across restarts
        rng = np.random.RandomState(42)
        self._action_dirs = {}
        for action in ["amplify", "dampen", "ground", "explore", "realign", "sustain", "wait"]:
            v = rng.randn(self.DIM).astype(np.float64)
            v /= np.linalg.norm(v)
            self._action_dirs[action] = v

    def imprint(self, action, strength, effect):
        """
        a decision happened. imprint it on gamma.

        action:   what was decided (dampen, explore, etc.)
        strength: how strong (0-1)
        effect:   field_entropy_after - field_entropy_before

        harmonic weighting: low frequencies persist, high frequencies decay.
        this creates a gamma that encodes both long-term style and recent behavior.
        """
        self._step += 1
        t = self._step
        direction = self._action_dirs.get(action, np.zeros(self.DIM))

        # harmonic weight — superposition of frequencies
        weight = 0.0
        for k in range(1, self.N_HARMONICS + 1):
            decay = np.exp(-0.01 * k)  # higher freq decays faster per step
            weight += np.cos(2 * np.pi * k * t / 64.0) * decay / k

        scale = strength * (1.0 + abs(effect)) * weight
        self.gamma += scale * direction

        # let magnitude grow (it's meaningful — how much personality has formed)
        # but cap to prevent explosion
        mag = np.linalg.norm(self.gamma)
        if mag > 1e-8:
            self.magnitude = float(mag)
            if mag > 10.0:
                self.gamma *= 10.0 / mag
                self.magnitude = 10.0

    def cosine_with(self, other_gamma):
        """cosine similarity with an organism's gamma vector."""
        if self.magnitude < 1e-8:
            return 0.0
        other = np.asarray(other_gamma, dtype=np.float64)
        other_mag = np.linalg.norm(other)
        if other_mag < 1e-8:
            return 0.0
        dim = min(len(self.gamma), len(other))
        return float(np.dot(self.gamma[:dim], other[:dim]) / (self.magnitude * other_mag))

    def direction(self):
        """unit vector, or zero if no personality yet."""
        if self.magnitude < 1e-8:
            return np.zeros(self.DIM)
        return self.gamma / self.magnitude

    def as_blob(self):
        """serialize for mesh.db."""
        return self.gamma.tobytes()

    def dominant_tendency(self):
        """which action direction is gamma most aligned with?"""
        if self.magnitude < 1e-8:
            return "none", 0.0
        best_action, best_cos = "none", -1.0
        d = self.direction()
        for action, v in self._action_dirs.items():
            cos = float(np.dot(d, v))
            if cos > best_cos:
                best_cos = cos
                best_action = action
        return best_action, best_cos


# ═══════════════════════════════════════════════════════════════════════════════
# harmonic net — weightless neural network
# ═══════════════════════════════════════════════════════════════════════════════

class HarmonicNet:
    """
    Weightless harmonic network. The neural network that has no weights.

    Architecture (3 layers, all computed from data):
        Layer 1: harmonic embedding — decompose entropy history into frequencies
        Layer 2: correlation matrix — pairwise gamma cosines between organisms
        Layer 3: phase aggregation — combine resonance + harmonics into steering

    "Weights" = organism correlations. They change every step.
    No backprop. No gradients. No training. Just resonance.

    Input:  field state (organisms, entropy, coherence, syntropy)
    Output: action_bias (per-action scores), strength_mod (confidence)
    """

    N_HARMONICS = 8

    def __init__(self, dim=32, lib=None):
        self.dim = dim
        self.lib = lib  # C library reference (for HarmonicNet C acceleration)
        self._entropy_history = []
        self._max_history = 64
        self._last_harmonics = np.zeros(self.N_HARMONICS)
        self._last_resonance = []

    def forward(self, organisms, field_entropy, field_coherence, field_syntropy, step):
        """
        One forward pass. No backprop ever.
        Uses C acceleration when available (libaml.so), falls back to numpy.

        Returns dict:
            action_bias: {action_name: score} — suggested emphasis
            strength_mod: 0-1 — confidence multiplier
            harmonics: frequency decomposition of entropy history
            resonance: per-organism resonance scores
            dominant_freq: which harmonic dominates
        """
        if not organisms:
            return {"action_bias": {}, "strength_mod": 0.0,
                    "harmonics": [], "resonance": [], "dominant_freq": 0}

        # ── C-accelerated path ──
        if self.lib is not None:
            return self._forward_c(organisms, field_entropy, step)

        # ── Python fallback path ──
        self._entropy_history.append(field_entropy)
        if len(self._entropy_history) > self._max_history:
            self._entropy_history = self._entropy_history[-self._max_history:]

        # ── Layer 1: Harmonic embedding ──
        # Fourier decomposition of entropy history
        harmonics = np.zeros(self.N_HARMONICS)
        T = len(self._entropy_history)
        if T >= 4:
            signal = np.array(self._entropy_history)
            for k in range(self.N_HARMONICS):
                t = np.arange(T, dtype=np.float64)
                harmonics[k] = np.sum(signal * np.sin(2 * np.pi * (k + 1) * t / T)) / T
        self._last_harmonics = harmonics

        # ── Layer 2: Correlation matrix ──
        # Pairwise gamma cosines — the "weight matrix"
        n = len(organisms)
        gammas = []
        for o in organisms:
            if o.gamma_direction and len(o.gamma_direction) >= self.dim * 8:
                g = np.frombuffer(o.gamma_direction[:self.dim * 8], dtype=np.float64)
                gammas.append(g[:self.dim] if len(g) >= self.dim else np.pad(g, (0, self.dim - len(g))))
            else:
                gammas.append(np.zeros(self.dim))
        gammas = np.array(gammas)

        norms = np.linalg.norm(gammas, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = gammas / norms
        corr = normed @ normed.T  # n×n correlation matrix

        # ── Layer 3: Phase aggregation ──
        # Each organism's "phase" = entropy relative to field mean
        entropies = np.array([o.entropy for o in organisms])
        mean_ent = np.mean(entropies) if len(entropies) > 0 else 1.0
        phases = entropies - mean_ent

        # Resonance: organisms that correlate AND have similar phase
        resonance = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    resonance[i] += corr[i, j] * np.exp(-abs(phases[i] - phases[j]))
            if n > 1:
                resonance[i] /= (n - 1)
        self._last_resonance = resonance.tolist()

        # ── Output: action biases from harmonic + resonance analysis ──
        action_bias = {}
        dominant_freq = int(np.argmax(np.abs(harmonics))) if T >= 4 else 0
        dominant_amp = float(harmonics[dominant_freq]) if T >= 4 else 0.0

        # Slow trend (low frequency dominant)
        if dominant_freq <= 1:
            if dominant_amp > 0.1:
                action_bias["dampen"] = min(dominant_amp * 2, 1.0)
            elif dominant_amp < -0.1:
                action_bias["amplify"] = min(abs(dominant_amp) * 2, 1.0)

        # Fast oscillation (high frequency) → ground
        elif dominant_freq >= 4:
            action_bias["ground"] = min(abs(float(harmonics[dominant_freq])) * 3, 1.0)

        # Low resonance → organisms aren't talking → explore
        mean_res = float(np.mean(resonance)) if len(resonance) > 0 else 0.0
        if mean_res < 0.3:
            action_bias["explore"] = max(action_bias.get("explore", 0), 0.5)

        # High resonance variance → some connected, some not → realign
        if n > 1:
            res_var = float(np.var(resonance))
            if res_var > 0.1:
                action_bias["realign"] = min(res_var * 3, 1.0)

        # Strength modulation: more data = more confident
        confidence = min(1.0, T / 16.0) * min(1.0, n / 4.0)
        strength_mod = 0.3 + 0.7 * confidence

        return {
            "action_bias": action_bias,
            "strength_mod": float(strength_mod),
            "harmonics": harmonics.tolist(),
            "resonance": resonance.tolist(),
            "dominant_freq": dominant_freq,
        }


    def _forward_c(self, organisms, field_entropy, step):
        """C-accelerated forward pass via libaml.so."""
        import ctypes

        # Push entropy to C circular buffer
        self.lib.am_harmonic_push_entropy(ctypes.c_float(field_entropy))

        # Also keep Python history for _entropy_history (used by tests/REPL)
        self._entropy_history.append(field_entropy)
        if len(self._entropy_history) > self._max_history:
            self._entropy_history = self._entropy_history[-self._max_history:]

        # Clear and push organism gammas
        self.lib.am_harmonic_clear()
        for o in organisms:
            gamma_f32 = None
            if o.gamma_direction and len(o.gamma_direction) >= self.dim * 8:
                g64 = np.frombuffer(o.gamma_direction[:self.dim * 8], dtype=np.float64)
                gamma_f32 = g64[:self.dim].astype(np.float32)
            elif o.gamma_direction and len(o.gamma_direction) >= self.dim * 4:
                gamma_f32 = np.frombuffer(o.gamma_direction[:self.dim * 4], dtype=np.float32)

            if gamma_f32 is not None and len(gamma_f32) > 0:
                arr = (ctypes.c_float * len(gamma_f32))(*gamma_f32)
                oid = hash(o.id) & 0x7FFFFFFF if isinstance(o.id, str) else int(o.id)
                self.lib.am_harmonic_push_gamma(
                    oid, arr, len(gamma_f32), ctypes.c_float(o.entropy))
            else:
                # Push zero gamma
                arr = (ctypes.c_float * self.dim)(*([0.0] * self.dim))
                oid = hash(o.id) & 0x7FFFFFFF if isinstance(o.id, str) else int(o.id)
                self.lib.am_harmonic_push_gamma(
                    oid, arr, self.dim, ctypes.c_float(o.entropy))

        # Forward pass in C
        result = self.lib.am_harmonic_forward(step)

        # Unpack C result
        harmonics = [float(result.harmonics[k]) for k in range(8)]
        n = result.n_organisms
        resonance = [float(result.resonance[i]) for i in range(n)]
        strength_mod = float(result.strength_mod)
        dominant_freq = int(result.dominant_freq)

        self._last_harmonics = np.array(harmonics)
        self._last_resonance = resonance

        # ── Action biases (same logic as Python, but using C-computed values) ──
        action_bias = {}
        if len(self._entropy_history) >= 4:
            dominant_amp = harmonics[dominant_freq]

            if dominant_freq <= 1:
                if dominant_amp > 0.1:
                    action_bias["dampen"] = min(dominant_amp * 2, 1.0)
                elif dominant_amp < -0.1:
                    action_bias["amplify"] = min(abs(dominant_amp) * 2, 1.0)
            elif dominant_freq >= 4:
                action_bias["ground"] = min(abs(harmonics[dominant_freq]) * 3, 1.0)

        mean_res = np.mean(resonance) if resonance else 0.0
        if mean_res < 0.3:
            action_bias["explore"] = max(action_bias.get("explore", 0), 0.5)

        if n > 1 and resonance:
            res_var = float(np.var(resonance))
            if res_var > 0.1:
                action_bias["realign"] = min(res_var * 3, 1.0)

        return {
            "action_bias": action_bias,
            "strength_mod": strength_mod,
            "harmonics": harmonics,
            "resonance": resonance,
            "dominant_freq": dominant_freq,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# mycelium syntropy — mathematical self-awareness of the orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class MyceliumSyntropy:
    """
    SyntropyTracker for mycelium.

    An organism's SyntropyTracker asks: "am I learning?"
    Mycelium's asks: "am I helping?"

    Tracks:
    - decision effectiveness: did my steering improve the field?
    - decision entropy: am I stuck in a pattern? (low = repetitive, high = random)
    - syntropy trend: is the field organizing under my guidance?
    - purpose magnitude: how strong is my steering direction?
    - purpose alignment: am I consistent? (low variance = aligned)
    """

    def __init__(self, window=16):
        self.window = window
        self.decision_history = []    # [{action, strength, delta_h, delta_s, delta_c, ...}]
        self.entropy_history = []     # field entropy over time
        self.syntropy_trend = 0.0     # positive = field organizing under my guidance
        self.decision_entropy = 0.0   # diversity of my decisions
        self.purpose_magnitude = 0.0  # how strongly am I steering
        self.purpose_alignment = 0.0  # how consistently
        self.effectiveness = {}       # action → mean improvement score
        self.last_action = "none"
        self._last_field = None       # snapshot before decision

    def snapshot_before(self, field_entropy, field_syntropy, field_coherence):
        """snapshot before making a decision — so we can measure effect."""
        self._last_field = {
            "entropy": field_entropy,
            "syntropy": field_syntropy,
            "coherence": field_coherence,
        }

    def record_decision(self, action, strength, field_entropy, field_syntropy, field_coherence):
        """record decision outcome — what happened to the field after my action?"""
        before = self._last_field or {
            "entropy": field_entropy,
            "syntropy": field_syntropy,
            "coherence": field_coherence,
        }

        record = {
            "action": action,
            "strength": strength,
            "delta_h": field_entropy - before["entropy"],
            "delta_s": field_syntropy - before["syntropy"],
            "delta_c": field_coherence - before["coherence"],
            "field_h": field_entropy,
            "field_s": field_syntropy,
            "field_c": field_coherence,
        }
        self.decision_history.append(record)
        if len(self.decision_history) > 64:
            self.decision_history = self.decision_history[-64:]

        self.entropy_history.append(field_entropy)
        if len(self.entropy_history) > 64:
            self.entropy_history = self.entropy_history[-64:]

        self.last_action = action
        self._last_field = {
            "entropy": field_entropy,
            "syntropy": field_syntropy,
            "coherence": field_coherence,
        }
        self._recompute()

    def _recompute(self):
        """recompute all syntropy metrics."""
        # 1. Syntropy trend: field entropy trending down = organizing
        if len(self.entropy_history) >= 4:
            half = len(self.entropy_history) // 2
            old_mean = float(np.mean(self.entropy_history[:half]))
            new_mean = float(np.mean(self.entropy_history[half:]))
            self.syntropy_trend = old_mean - new_mean

        # 2. Decision entropy: diversity of recent actions
        if len(self.decision_history) >= 4:
            recent = self.decision_history[-min(self.window, len(self.decision_history)):]
            counts = {}
            for d in recent:
                a = d["action"]
                counts[a] = counts.get(a, 0) + 1
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            self.decision_entropy = float(-sum(p * np.log(p + 1e-10) for p in probs))

        # 3. Per-action effectiveness
        eff_raw = {}
        for d in self.decision_history:
            a = d["action"]
            if a not in eff_raw:
                eff_raw[a] = []
            score = d["delta_s"] + d["delta_c"] * 0.5
            eff_raw[a].append(score)
        self.effectiveness = {}
        for a, vals in eff_raw.items():
            self.effectiveness[a] = float(np.mean(vals[-self.window:]))

        # 4. Purpose: how strongly and consistently am I steering
        if len(self.decision_history) >= 2:
            deltas = [d["delta_s"] for d in self.decision_history[-self.window:]]
            self.purpose_magnitude = float(abs(np.mean(deltas)))
            self.purpose_alignment = float(1.0 / (1.0 + np.std(deltas))) if len(deltas) >= 2 else 0.0

    def measure(self):
        """current metrics as dict."""
        return {
            "syntropy_trend": round(self.syntropy_trend, 4),
            "decision_entropy": round(self.decision_entropy, 4),
            "purpose_magnitude": round(self.purpose_magnitude, 4),
            "purpose_alignment": round(self.purpose_alignment, 4),
            "effectiveness": {k: round(v, 4) for k, v in self.effectiveness.items()},
            "last_action": self.last_action,
            "n_decisions": len(self.decision_history),
        }

    def should_change_strategy(self):
        """am I stuck or failing? should I try something different?"""
        if len(self.decision_history) < 8:
            return False, "too early"

        if self.decision_entropy < 0.5:
            return True, f"stuck in pattern (H_dec={self.decision_entropy:.2f})"

        if self.syntropy_trend < -0.05 and len(self.entropy_history) >= 8:
            return True, f"field dissolving under guidance (trend={self.syntropy_trend:.3f})"

        return False, "on track"

    def status_line(self):
        """one-line self-report."""
        m = self.measure()
        return (f"syntropy_trend={m['syntropy_trend']:+.3f} "
                f"H_dec={m['decision_entropy']:.2f} "
                f"purpose={m['purpose_magnitude']:.3f}×{m['purpose_alignment']:.2f} "
                f"n={m['n_decisions']}")


# ═══════════════════════════════════════════════════════════════════════════════
# mycelium core
# ═══════════════════════════════════════════════════════════════════════════════

class Mycelium:
    """the orchestrator. connects organisms through METHOD."""

    def __init__(self, mesh_path="mesh.db", interval=1.0, verbose=True):
        self.method = Method(mesh_path)
        self.mesh_path = mesh_path
        self.interval = interval
        self.verbose = verbose
        self.monitor = FieldMonitor()
        self.drift = DriftTracker()
        self.voice = MyceliumVoice()
        self.gamma = MyceliumGamma()
        self.harmonic = HarmonicNet(lib=self.method.lib)
        self.syntropy = MyceliumSyntropy()
        self.pulse = FieldPulse()
        self.dissonance = SteeringDissonance()
        self.attention = OrganismAttention()
        self.running = False
        self._step_count = 0
        self._last_steering = None
        # SACRED: field lock. Only ONE step at a time.
        # Sequential field evolution = coherence. (from leo async architecture)
        self._field_lock = asyncio.Lock()
        # Sentinel — DNA watcher (scans every 5 steps to avoid I/O spam)
        dna_path = os.path.join(os.path.dirname(os.path.abspath(mesh_path)), "dna")
        aml_path = os.path.join(dna_path, "sentinel.aml") if os.path.isdir(dna_path) else None
        self.sentinel = Sentinel(
            dna_path=dna_path,
            aml_path=aml_path if aml_path and os.path.exists(aml_path) else None,
            lib=self.method.lib,
        ) if os.path.isdir(dna_path) else None
        self._sentinel_interval = 5  # scan every N steps

    def step(self):
        """one tick: snapshot → METHOD → harmonic → gamma → syntropy → steer."""
        # 1. Snapshot field state BEFORE decision
        self.method.read_field()
        pre_h = self.method.field_entropy()
        pre_s = self.method.field_syntropy()
        pre_c = self.method.field_coherence()
        self.syntropy.snapshot_before(pre_h, pre_s, pre_c)

        # 2. METHOD computes raw steering (C-native, BLAS)
        steering = self.method.step(dt=self.interval)

        # 2b. Measure field pulse (harmonix-inspired)
        self.pulse.measure(self.method.organisms, pre_h)

        # 2c. Pre-steering organism entropies (for attention tracking)
        pre_entropies = {o.id: o.entropy for o in self.method.organisms}

        # 3. HarmonicNet refines — looks at deeper patterns
        h_out = self.harmonic.forward(
            self.method.organisms,
            steering.get("entropy", 0),
            steering.get("coherence", 1),
            steering.get("syntropy", 0),
            self._step_count,
        )

        # Apply harmonic refinement to strength
        if h_out["strength_mod"] > 0:
            base_strength = steering.get("strength", 0.5)
            steering["strength"] = base_strength * h_out["strength_mod"]

        # If harmonic net suggests a different action strongly, note it
        if h_out["action_bias"]:
            best_harmonic = max(h_out["action_bias"], key=h_out["action_bias"].get)
            steering["harmonic_suggestion"] = best_harmonic
            steering["harmonic_confidence"] = h_out["action_bias"][best_harmonic]

        # 3b. Dissonance modulation (harmonix principle: intent vs outcome)
        post_h = steering.get("entropy", pre_h)
        post_s = steering.get("syntropy", pre_s)
        post_c = steering.get("coherence", pre_c)
        action = steering.get("action", "wait")
        dis = self.dissonance.update(action, post_h - pre_h, post_c - pre_c)
        steering["strength"] = min(1.0,
            steering.get("strength", 0.5) * self.dissonance.strength_multiplier())

        # 4. Write steering to mesh.db for Rust
        self.method.write_steering(steering)

        # 4b. Update organism attention (harmonix cloud morphing)
        self.attention.update(self.method.organisms, action=steering.get("action", "wait"),
                              pre_entropies=pre_entropies)

        # 5. Record decision in syntropy tracker
        strength = steering.get("strength", 0)
        self.syntropy.record_decision(action, strength, post_h, post_s, post_c)

        # 6. Imprint on gamma
        effect = post_h - pre_h
        self.gamma.imprint(action, strength, effect)

        # 7. Monitor + bookkeeping
        self.monitor.record(steering)
        self._step_count += 1
        self._last_steering = steering

        # Add self-awareness to steering dict
        steering["gamma_magnitude"] = self.gamma.magnitude
        tendency, tendency_cos = self.gamma.dominant_tendency()
        steering["gamma_tendency"] = tendency
        steering["syntropy_trend"] = self.syntropy.syntropy_trend
        steering["decision_entropy"] = self.syntropy.decision_entropy
        steering["dissonance"] = self.dissonance.dissonance
        steering["pulse_novelty"] = self.pulse.novelty
        steering["pulse_arousal"] = self.pulse.arousal

        # Check if we should change strategy
        should_change, reason = self.syntropy.should_change_strategy()
        if should_change:
            steering["strategy_warning"] = reason

        # Check for drifters every 4 steps
        drift_report = None
        if self._step_count % 4 == 0:
            self.drift.update(self.method)
            drift_report = self.drift.report()

        # Sentinel: scan DNA directory for changes
        sentinel_report = None
        if self.sentinel and self._step_count % self._sentinel_interval == 0:
            changes = self.sentinel.scan()
            if changes:
                sentinel_report = self.sentinel.report()
                steering["dna_changes"] = len(changes)

        if self.verbose:
            status = self.monitor.status_line(steering)
            # Append gamma info
            if self.gamma.magnitude > 0.01:
                status += f" γ={self.gamma.magnitude:.2f}({tendency})"
            # Pulse + dissonance
            p = self.pulse
            if p.arousal > 0.01 or p.novelty > 0.01:
                status += f" pulse(n={p.novelty:.1f},a={p.arousal:.1f},e={p.entropy:.1f})"
            if self.dissonance.dissonance > 0.01:
                status += f" d={self.dissonance.dissonance:.2f}"
            # Append syntropy self-check
            if self.syntropy.syntropy_trend != 0:
                status += f" Σ={self.syntropy.syntropy_trend:+.3f}"
            if should_change:
                status += f"  ⚠ {reason}"
            print(status)
            if drift_report:
                print(drift_report)
            if sentinel_report:
                print(sentinel_report)

        return steering

    async def async_step(self):
        """async one tick: snapshot → METHOD → harmonic → gamma → syntropy → steer.
        Uses asyncio.Lock for field coherence (from leo async architecture).
        Uses aiosqlite for non-blocking SQLite I/O.
        C computation stays sync (0.7μs — not worth async overhead)."""
        async with self._field_lock:
            # 1. Snapshot field state BEFORE decision
            self.method.read_field()  # sync C calls (fast)
            pre_h = self.method.field_entropy()
            pre_s = self.method.field_syntropy()
            pre_c = self.method.field_coherence()
            self.syntropy.snapshot_before(pre_h, pre_s, pre_c)

            # 2. METHOD computes raw steering (C-native, BLAS) — sync, fast
            steering = self.method.step(dt=self.interval)

            # 2b. Measure field pulse
            self.pulse.measure(self.method.organisms, pre_h)
            pre_entropies = {o.id: o.entropy for o in self.method.organisms}

            # 3. HarmonicNet refines
            h_out = self.harmonic.forward(
                self.method.organisms,
                steering.get("entropy", 0),
                steering.get("coherence", 1),
                steering.get("syntropy", 0),
                self._step_count,
            )
            if h_out["strength_mod"] > 0:
                steering["strength"] = steering.get("strength", 0.5) * h_out["strength_mod"]

            if h_out["action_bias"]:
                best_harmonic = max(h_out["action_bias"], key=h_out["action_bias"].get)
                steering["harmonic_suggestion"] = best_harmonic
                steering["harmonic_confidence"] = h_out["action_bias"][best_harmonic]

            # 3b. Dissonance modulation
            post_h = steering.get("entropy", pre_h)
            post_s = steering.get("syntropy", pre_s)
            post_c = steering.get("coherence", pre_c)
            action = steering.get("action", "wait")
            dis = self.dissonance.update(action, post_h - pre_h, post_c - pre_c)
            steering["strength"] = min(1.0,
                steering.get("strength", 0.5) * self.dissonance.strength_multiplier())

            # 4. Write steering to mesh.db — ASYNC
            await self._async_write_steering(steering)

            # 4b. Organism attention
            self.attention.update(self.method.organisms, action=action,
                                  pre_entropies=pre_entropies)

            # 5. Record decision in syntropy tracker
            strength = steering.get("strength", 0)
            self.syntropy.record_decision(action, strength, post_h, post_s, post_c)

            # 6. Imprint on gamma
            self.gamma.imprint(action, strength, post_h - pre_h)

            # 7. Monitor + bookkeeping
            self.monitor.record(steering)
            self._step_count += 1
            self._last_steering = steering

            # Self-awareness metadata
            steering["gamma_magnitude"] = self.gamma.magnitude
            tendency, tendency_cos = self.gamma.dominant_tendency()
            steering["gamma_tendency"] = tendency
            steering["syntropy_trend"] = self.syntropy.syntropy_trend
            steering["decision_entropy"] = self.syntropy.decision_entropy
            steering["dissonance"] = self.dissonance.dissonance
            steering["pulse_novelty"] = self.pulse.novelty
            steering["pulse_arousal"] = self.pulse.arousal

            should_change, reason = self.syntropy.should_change_strategy()
            if should_change:
                steering["strategy_warning"] = reason

            # Drifters
            drift_report = None
            if self._step_count % 4 == 0:
                self.drift.update(self.method)
                drift_report = self.drift.report()

            if self.verbose:
                status = self.monitor.status_line(steering)
                if self.gamma.magnitude > 0.01:
                    status += f" γ={self.gamma.magnitude:.2f}({tendency})"
                p = self.pulse
                if p.arousal > 0.01 or p.novelty > 0.01:
                    status += f" pulse(n={p.novelty:.1f},a={p.arousal:.1f},e={p.entropy:.1f})"
                if self.dissonance.dissonance > 0.01:
                    status += f" d={self.dissonance.dissonance:.2f}"
                if self.syntropy.syntropy_trend != 0:
                    status += f" Σ={self.syntropy.syntropy_trend:+.3f}"
                if should_change:
                    status += f"  ⚠ {reason}"
                print(status)
                if drift_report:
                    print(drift_report)

            return steering

    async def _async_write_steering(self, steering):
        """write steering to mesh.db using aiosqlite (non-blocking)."""
        if aiosqlite is None:
            # Fallback to sync
            self.method.write_steering(steering)
            return
        try:
            async with aiosqlite.connect(self.mesh_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("""
                    INSERT OR REPLACE INTO field_steering
                    (id, action, strength, target_id, entropy, syntropy,
                     coherence, trend, n_organisms, updated_at)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    steering.get("action", "wait"),
                    steering.get("strength", 0.0),
                    str(steering.get("target", "")),
                    steering.get("entropy", 0.0),
                    steering.get("syntropy", 0.0),
                    steering.get("coherence", 0.0),
                    steering.get("trend", 0.0),
                    steering.get("n_organisms", 0),
                    time.time(),
                ))
                await db.commit()
        except Exception as e:
            # Fallback to sync on error
            self.method.write_steering(steering)

    async def async_read_field(self):
        """read organisms from mesh.db using aiosqlite, push to C METHOD."""
        if aiosqlite is None:
            self.method.read_field()
            return self.method.organisms
        try:
            async with aiosqlite.connect(self.mesh_path) as db:
                async with db.execute("""
                    SELECT id, pid, stage, n_params, syntropy, entropy,
                           gamma_direction, gamma_magnitude, last_heartbeat,
                           element
                    FROM organisms
                    WHERE status = 'alive' AND last_heartbeat > ?
                """, (time.time() - 120,)) as cursor:
                    rows = await cursor.fetchall()
            # Parse organisms (sync — fast)
            from ariannamethod.method import Organism
            self.method.organisms = [Organism(row) for row in rows]
            # Push to C (sync — 0.7μs)
            if self.method.lib is not None:
                import ctypes
                self.method.lib.am_method_clear()
                gammas = {}
                for o in self.method.organisms:
                    if o.gamma_direction and len(o.gamma_direction) > 0:
                        try:
                            arr = np.frombuffer(o.gamma_direction, dtype=np.float64)
                            norm = np.linalg.norm(arr)
                            if norm > 1e-12:
                                gammas[o.id] = arr / norm
                        except Exception:
                            pass
                if len(gammas) >= 2:
                    vecs = list(gammas.values())
                    min_len = min(len(v) for v in vecs)
                    mean_gamma = np.mean([v[:min_len] for v in vecs], axis=0)
                    mean_norm = np.linalg.norm(mean_gamma)
                    mean_gamma = mean_gamma / mean_norm if mean_norm > 1e-12 else None
                else:
                    mean_gamma = None
                for o in self.method.organisms:
                    gamma_mag, gamma_cos = 0.0, 0.0
                    if o.id in gammas:
                        g = gammas[o.id]
                        gamma_mag = float(np.linalg.norm(g))
                        if mean_gamma is not None:
                            ml = min(len(g), len(mean_gamma))
                            gamma_cos = float(np.dot(g[:ml], mean_gamma[:ml]))
                    oid = hash(o.id) & 0x7FFFFFFF if isinstance(o.id, str) else int(o.id)
                    self.method.lib.am_method_push_organism(
                        oid, ctypes.c_float(o.entropy), ctypes.c_float(o.syntropy),
                        ctypes.c_float(gamma_mag), ctypes.c_float(gamma_cos))
        except Exception:
            self.method.read_field()
        return self.method.organisms

    def speak(self):
        """field report in mycelium's voice."""
        if self._last_steering is None:
            self.step()
        drifters = self.drift.drifters if self.drift.drifters else None
        return self.voice.speak(
            self._last_steering,
            self.method.organisms,
            drifters,
        )

    # ── REPL commands ──

    def cmd_field(self):
        """full field report."""
        self.step()
        self.drift.update(self.method)
        return self.speak()

    def cmd_who(self):
        """list all organisms."""
        self.method.read_field()
        if not self.method.organisms:
            return "no organisms alive."
        lines = []
        for o in self.method.organisms:
            age = time.time() - o.last_seen if o.last_seen else 0
            elem_tag = f" [{o.element}]" if o.element else ""
            lines.append(
                f"  {o.id}{elem_tag}: stage={o.stage} entropy={o.entropy:.2f} "
                f"syntropy={o.syntropy:.2f} params={o.n_params} "
                f"last_seen={age:.0f}s ago"
            )
        return f"{len(self.method.organisms)} organisms:\n" + "\n".join(lines)

    def cmd_drift(self):
        """show drifters."""
        self.method.read_field()
        self.drift.update(self.method)
        report = self.drift.report()
        return report if report else "no drifters. field is stable."

    def cmd_status(self):
        """one-line status."""
        self.step()
        return self.monitor.status_line(self._last_steering)

    def cmd_entropy(self):
        """field entropy details."""
        self.method.read_field()
        if not self.method.organisms:
            return "no organisms."
        ent = self.method.field_entropy()
        lines = [f"field entropy: {ent:.4f}"]
        for o in self.method.organisms:
            bar = "#" * int(o.entropy * 10)
            lines.append(f"  {o.id}: {o.entropy:.3f} {bar}")
        return "\n".join(lines)

    def cmd_coherence(self):
        """field coherence details."""
        self.method.read_field()
        coh = self.method.field_coherence()
        return f"field coherence: {coh:.4f}"

    def cmd_gamma(self):
        """mycelium's personality vector."""
        lines = [f"γ_myc magnitude: {self.gamma.magnitude:.4f}"]
        tendency, cos = self.gamma.dominant_tendency()
        lines.append(f"dominant tendency: {tendency} (cos={cos:.3f})")

        # Compare with each organism's gamma
        self.method.read_field()
        if self.method.organisms:
            lines.append("alignment with organisms:")
            for o in self.method.organisms:
                if o.gamma_direction and len(o.gamma_direction) >= 8:
                    g = np.frombuffer(o.gamma_direction[:self.gamma.DIM * 8], dtype=np.float64)
                    cos_val = self.gamma.cosine_with(g)
                    bar = "#" * int(abs(cos_val) * 20)
                    sign = "+" if cos_val >= 0 else "-"
                    lines.append(f"  {o.id}: {sign}{abs(cos_val):.3f} {bar}")

        return "\n".join(lines)

    def cmd_syntropy(self):
        """mycelium's self-awareness report."""
        m = self.syntropy.measure()
        lines = [
            f"syntropy trend: {m['syntropy_trend']:+.4f}"
            + (" (organizing)" if m['syntropy_trend'] > 0.01 else
               " (dissolving)" if m['syntropy_trend'] < -0.01 else " (stable)"),
            f"decision entropy: {m['decision_entropy']:.4f}"
            + (" (diverse)" if m['decision_entropy'] > 1.0 else
               " (repetitive)" if m['decision_entropy'] < 0.5 else " (balanced)"),
            f"purpose: magnitude={m['purpose_magnitude']:.4f} alignment={m['purpose_alignment']:.4f}",
            f"decisions: {m['n_decisions']}",
        ]
        if m['effectiveness']:
            lines.append("effectiveness per action:")
            for a, score in sorted(m['effectiveness'].items(), key=lambda x: -x[1]):
                bar = "#" * int(max(0, score) * 20)
                lines.append(f"  {a}: {score:+.4f} {bar}")

        should_change, reason = self.syntropy.should_change_strategy()
        if should_change:
            lines.append(f"⚠ strategy change needed: {reason}")
        else:
            lines.append(f"strategy: {reason}")

        return "\n".join(lines)

    def cmd_pulse(self):
        """field pulse (harmonix-style)."""
        p = self.pulse
        lines = [
            f"novelty:  {p.novelty:.4f}" + (" (new organisms!)" if p.novelty > 0.5 else ""),
            f"arousal:  {p.arousal:.4f}" + (" (field is volatile)" if p.arousal > 0.5 else ""),
            f"entropy:  {p.entropy:.4f}" + (" (diverse)" if p.entropy > 1.0 else " (uniform)" if p.entropy < 0.3 else ""),
        ]
        d = self.dissonance
        lines.append(f"dissonance: {d.dissonance:.4f} → strength ×{d.strength_multiplier():.2f}")
        if d.history:
            lines.append("recent:")
            for h in d.history[-5:]:
                lines.append(f"  {h['action']}: raw={h['raw']:.3f} ema={h['ema']:.3f}")
        return "\n".join(lines)

    def cmd_attention(self):
        """organism attention map."""
        lines = ["organism attention (who responds to steering?):"]
        lines.append(self.attention.report())
        top = self.attention.top_organisms(3)
        if top:
            lines.append(f"most responsive: {', '.join(f'{oid}({w:.2f})' for oid, w in top)}")
        return "\n".join(lines)

    def cmd_harmonics(self):
        """harmonic net state — frequency decomposition."""
        h = self.harmonic._last_harmonics
        lines = ["entropy harmonics (frequency decomposition):"]
        for k in range(len(h)):
            bar_len = int(abs(h[k]) * 40)
            sign = "+" if h[k] >= 0 else "-"
            bar = "#" * bar_len
            lines.append(f"  f{k+1}: {sign}{abs(h[k]):.4f} {bar}")

        if self.harmonic._last_resonance:
            lines.append("organism resonance:")
            orgs = self.method.organisms if self.method.organisms else []
            for i, r in enumerate(self.harmonic._last_resonance):
                name = orgs[i].id if i < len(orgs) else f"org-{i}"
                bar = "#" * int(abs(r) * 20)
                lines.append(f"  {name}: {r:.3f} {bar}")

        return "\n".join(lines)

    def cmd_sentinel(self):
        """sentinel status and latest changes."""
        if not self.sentinel:
            return "[sentinel] not active (no dna/ directory)."
        lines = [self.sentinel.status_line()]
        changes = self.sentinel.scan()
        if changes:
            lines.append(self.sentinel.report())
        else:
            lines.append("[sentinel] no changes since last scan.")
        return "\n".join(lines)

    def cmd_dna(self):
        """DNA directory overview."""
        if not self.sentinel:
            return "[dna] not active (no dna/ directory)."
        dna = self.sentinel.dna_path
        lines = [f"[dna] root: {dna}"]
        for zone in ("incoming", "shared", "output"):
            zpath = os.path.join(dna, zone)
            if os.path.isdir(zpath):
                count = 0
                total_size = 0
                for root, dirs, files in os.walk(zpath):
                    for f in files:
                        if not f.startswith('.'):
                            count += 1
                            total_size += os.path.getsize(os.path.join(root, f))
                lines.append(f"  {zone}/: {count} files, {total_size:,} bytes")
            else:
                lines.append(f"  {zone}/: (not found)")
        lines.append(f"  watched: {self.sentinel.watched_count()} files total")
        return "\n".join(lines)

    def cmd_help(self):
        return (
            "mycelium commands:\n"
            "  /field     — full field report (voice)\n"
            "  /who       — list organisms\n"
            "  /drift     — show drifters\n"
            "  /status    — one-line status\n"
            "  /entropy   — entropy per organism\n"
            "  /coherence — pairwise gamma coherence\n"
            "  /gamma     — mycelium's personality vector\n"
            "  /syntropy  — self-awareness: am I helping?\n"
            "  /harmonics — frequency decomposition of field\n"
            "  /pulse     — field pulse (novelty, arousal, entropy)\n"
            "  /attention — organism attention map\n"
            "  /sentinel  — DNA watcher status + changes\n"
            "  /dna       — DNA directory overview\n"
            "  /step      — force one METHOD step\n"
            "  /json      — last steering as JSON\n"
            "  /quit      — exit\n"
            "\n"
            "or just type anything — mycelium will read the field and respond."
        )

    # ── daemon mode ──

    async def run_daemon(self):
        """async background loop: async_step every interval, no REPL."""
        self.running = True
        lib_status = "C+BLAS" if self.method.lib else "Python fallback"
        async_status = "aiosqlite" if aiosqlite else "sync fallback"
        print(f"[mycelium] daemon started. METHOD: {lib_status}. I/O: {async_status}")
        print(f"[mycelium] mesh: {self.method.mesh_path}")
        print(f"[mycelium] interval: {self.interval}s")
        print()

        while self.running:
            try:
                await self.async_step()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[mycelium] error: {e}")

            await asyncio.sleep(self.interval)

        print("\n[mycelium] stopped.")

    def stop(self):
        self.running = False

    async def _bg_async_stepper(self):
        """background async stepper for REPL mode."""
        while self.running:
            try:
                await self.async_step()
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    # ── REPL ──

    def repl(self):
        """interactive REPL."""
        # Initial field read
        self.method.read_field()
        n = len(self.method.organisms)
        print(self.voice.greet(
            self.method.lib is not None,
            self.method.mesh_path,
            n,
        ))
        if self.sentinel:
            print(f"sentinel: watching {self.sentinel.dna_path} "
                  f"({self.sentinel.watched_count()} files)\n")

        # Background stepper — async task in event loop
        self.running = True
        self._bg_loop = asyncio.new_event_loop()

        def _run_bg_loop():
            asyncio.set_event_loop(self._bg_loop)
            self._bg_loop.run_until_complete(self._bg_async_stepper())

        bg = threading.Thread(target=_run_bg_loop, daemon=True)
        bg.start()

        try:
            while True:
                try:
                    line = input("mycelium> ").strip()
                except EOFError:
                    break

                if not line:
                    continue

                if line in ("/quit", "/exit", "quit", "exit"):
                    break
                elif line == "/field":
                    print(self.cmd_field())
                elif line == "/who":
                    print(self.cmd_who())
                elif line == "/drift":
                    print(self.cmd_drift())
                elif line == "/status":
                    print(self.cmd_status())
                elif line == "/entropy":
                    print(self.cmd_entropy())
                elif line == "/coherence":
                    print(self.cmd_coherence())
                elif line == "/gamma":
                    print(self.cmd_gamma())
                elif line == "/syntropy":
                    print(self.cmd_syntropy())
                elif line == "/harmonics":
                    print(self.cmd_harmonics())
                elif line == "/pulse":
                    print(self.cmd_pulse())
                elif line == "/attention":
                    print(self.cmd_attention())
                elif line == "/sentinel":
                    print(self.cmd_sentinel())
                elif line == "/dna":
                    print(self.cmd_dna())
                elif line == "/step":
                    s = self.step()
                    print(json.dumps(s, indent=2))
                elif line == "/json":
                    if self._last_steering:
                        print(json.dumps(self._last_steering, indent=2))
                    else:
                        print("no data yet. run /step first.")
                elif line == "/help":
                    print(self.cmd_help())
                else:
                    # Any other input: read field, speak
                    self.method.read_field()
                    self.drift.update(self.method)
                    if self._last_steering is None:
                        self.step()
                    print(self.speak())

                print()

        except KeyboardInterrupt:
            pass

        self.running = False
        print("\nmycelium sleeps.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="mycelium — distributed cognition orchestrator")
    parser.add_argument("--mesh", default="mesh.db",
                        help="path to mesh.db (default: ./mesh.db)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between METHOD steps (default: 1.0)")
    parser.add_argument("--once", action="store_true",
                        help="single step, print, exit")
    parser.add_argument("--daemon", action="store_true",
                        help="background daemon (no REPL)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-step output in daemon mode")
    args = parser.parse_args()

    myc = Mycelium(
        mesh_path=args.mesh,
        interval=args.interval,
        verbose=not args.quiet,
    )

    if args.once:
        steering = myc.step()
        print(json.dumps(steering, indent=2))
        return

    if args.daemon:
        def handle_signal(sig, frame):
            myc.stop()
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        asyncio.run(myc.run_daemon())
        return

    # Default: interactive REPL
    myc.repl()


if __name__ == "__main__":
    main()
