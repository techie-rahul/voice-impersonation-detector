"""Phase 2 risk engine for the SIH26104 voice-cloning detection project.

Pure-Python scoring. Given *already-computed* ML scores (AASIST spoof
probability + speaker-verification similarity) plus lightweight boolean call
context and a transaction amount, produce:

* a blended ``risk_score`` in ``[0, 100]``
* a ``risk_level``  -- ``"LOW"`` / ``"MEDIUM"`` / ``"HIGH"``
* a ``decision``    -- ``"ALLOW"`` / ``"VERIFY"`` / ``"BLOCK"``
* human-readable ``risk_factors``

This module deliberately has **no** dependency on FastAPI, PyTorch, AASIST,
Resemblyzer, any database, or frontend code, so it can be unit-tested and
reused in isolation. It never performs ML inference and never fabricates
scores -- callers must supply real values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

# --- Blend weights (final score) --------------------------------------------
# How much each component contributes to the final risk. These sum to 1.0.
W_SYNTHETIC: float = 0.50
W_SPEAKER_MISMATCH: float = 0.30
W_CONTEXT: float = 0.20

# --- Context signal weights ----------------------------------------------
# Relative importance of each boolean context signal. The context component is
# normalised by the *sum* of these weights so it always stays within [0, 1],
# no matter how many signals fire (one signal -> a fraction, all -> 1.0).
CONTEXT_WEIGHTS: dict[str, float] = {
    "transfer": 0.25,
    "otp": 0.25,
    "urgent": 0.15,
    "unknown_caller": 0.15,
    "high_value_transaction": 0.20,
}
_CONTEXT_WEIGHT_TOTAL: float = sum(CONTEXT_WEIGHTS.values())

# Boolean context keys accepted from the caller (``high_value_transaction`` is
# derived from ``transaction_amount``, not passed in ``context``).
CONTEXT_KEYS: tuple[str, ...] = ("transfer", "otp", "urgent", "unknown_caller")

# A transaction at or above this INR value is treated as "high value".
HIGH_VALUE_THRESHOLD_INR: float = 100_000

# Thresholds used ONLY to phrase risk_factors -- they do not change the score.
SYNTHETIC_FACTOR_THRESHOLD: float = 0.50   # p_spoof at/above this -> flagged
MISMATCH_FACTOR_THRESHOLD: float = 0.50    # (1 - speaker_match) at/above -> flagged

# Decision thresholds on the 0-100 risk score.
#   risk < 30           -> LOW    / ALLOW
#   30 <= risk <= 70    -> MEDIUM / VERIFY
#   risk > 70           -> HIGH   / BLOCK
LOW_RISK_MAX: float = 30.0
HIGH_RISK_MIN: float = 70.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the inclusive ``[low, high]`` range."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class RiskAssessment:
    """Structured result returned by :func:`calculate_risk`."""

    risk_score: float                              # 0-100, rounded to 2 dp
    risk_level: str                                # "LOW" | "MEDIUM" | "HIGH"
    decision: str                                  # "ALLOW" | "VERIFY" | "BLOCK"
    risk_factors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a plain ``dict`` copy (handy for JSON serialisation)."""
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "risk_factors": list(self.risk_factors),
        }


def calculate_risk(
    synthetic_score: float,
    speaker_match: float,
    context: Optional[Mapping[str, bool]],
    transaction_amount: float,
) -> RiskAssessment:
    """Blend precomputed ML scores and call context into a risk assessment.

    Args:
        synthetic_score: AASIST spoof probability (``p_spoof``), ``0..1``.
            Higher means more likely synthetic/cloned.
        speaker_match: Speaker-verification similarity, ``0..1``. Higher means
            more likely the same (enrolled) speaker.
        context: Mapping of boolean context signals. Recognised keys:
            ``"transfer"``, ``"otp"``, ``"urgent"``, ``"unknown_caller"``.
            Missing keys are treated as ``False``; unknown keys are ignored.
            ``None`` is accepted and means "no signals".
        transaction_amount: Transaction value in INR, must be ``>= 0``.

    Returns:
        A :class:`RiskAssessment`.

    Raises:
        ValueError: if ``synthetic_score`` or ``speaker_match`` is outside
            ``[0, 1]``, or ``transaction_amount`` is negative.
    """
    # --- validate inputs (no silent coercion of out-of-range values) -----
    if not 0.0 <= synthetic_score <= 1.0:
        raise ValueError(
            f"synthetic_score must be in [0, 1], got {synthetic_score!r}"
        )
    if not 0.0 <= speaker_match <= 1.0:
        raise ValueError(
            f"speaker_match must be in [0, 1], got {speaker_match!r}"
        )
    if transaction_amount < 0:
        raise ValueError(
            f"transaction_amount must be >= 0, got {transaction_amount!r}"
        )

    ctx = context or {}
    signals = {key: bool(ctx.get(key, False)) for key in CONTEXT_KEYS}
    high_value = transaction_amount >= HIGH_VALUE_THRESHOLD_INR

    # --- 1. synthetic voice risk ---------------------------------------
    # p_spoof is already a probability in [0, 1]; use it directly.
    synthetic_risk = synthetic_score

    # --- 2. speaker mismatch --------------------------------------------
    # Distance from a perfect speaker match, clamped for safety.
    speaker_mismatch = _clamp(1.0 - speaker_match)

    # --- 3. context risk ----------------------------------------------
    # Add the weight of every signal that fired, then normalise by the total
    # of all context weights so the component is bounded to [0, 1].
    active_weight = 0.0
    if signals["transfer"]:
        active_weight += CONTEXT_WEIGHTS["transfer"]
    if signals["otp"]:
        active_weight += CONTEXT_WEIGHTS["otp"]
    if signals["urgent"]:
        active_weight += CONTEXT_WEIGHTS["urgent"]
    if signals["unknown_caller"]:
        active_weight += CONTEXT_WEIGHTS["unknown_caller"]
    if high_value:
        active_weight += CONTEXT_WEIGHTS["high_value_transaction"]

    context_risk = _clamp(active_weight / _CONTEXT_WEIGHT_TOTAL)

    # --- 4. blended final risk, scaled to 0-100 -----------------------
    risk_fraction = (
        synthetic_risk * W_SYNTHETIC
        + speaker_mismatch * W_SPEAKER_MISMATCH
        + context_risk * W_CONTEXT
    )
    risk_score = round(_clamp(risk_fraction * 100.0, 0.0, 100.0), 2)

    # --- risk level + decision ---------------------------------------
    if risk_score < LOW_RISK_MAX:
        risk_level, decision = "LOW", "ALLOW"
    elif risk_score <= HIGH_RISK_MIN:
        risk_level, decision = "MEDIUM", "VERIFY"
    else:
        risk_level, decision = "HIGH", "BLOCK"

    # --- human-readable factors ------------------------------------
    factors: list[str] = []
    if synthetic_score >= SYNTHETIC_FACTOR_THRESHOLD:
        factors.append("High synthetic voice probability")
    if speaker_mismatch >= MISMATCH_FACTOR_THRESHOLD:
        factors.append("Speaker identity mismatch")
    if signals["transfer"]:
        factors.append("Money transfer requested")
    if signals["otp"]:
        factors.append("OTP request detected")
    if signals["urgent"]:
        factors.append("Urgent request detected")
    if signals["unknown_caller"]:
        factors.append("Unknown caller")
    if high_value:
        factors.append("High-value transaction")

    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        decision=decision,
        risk_factors=factors,
    )
