"""Unit tests for the pure-Python risk engine (``app.risk_engine``)."""

from __future__ import annotations

import pytest

from app.risk_engine import (
    HIGH_VALUE_THRESHOLD_INR,
    RiskAssessment,
    calculate_risk,
)


def risk(
    synthetic_score=0.0,
    speaker_match=1.0,
    context=None,
    transaction_amount=0.0,
) -> RiskAssessment:
    """Convenience wrapper with benign defaults (yields risk_score 0)."""
    return calculate_risk(synthetic_score, speaker_match, context, transaction_amount)


# --------------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------------

def test_result_is_structured():
    result = risk(synthetic_score=0.4, speaker_match=0.6)
    assert isinstance(result, RiskAssessment)
    assert isinstance(result.risk_score, float)
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert result.decision in {"ALLOW", "VERIFY", "BLOCK"}
    assert isinstance(result.risk_factors, list)
    assert all(isinstance(f, str) for f in result.risk_factors)
    assert set(result.as_dict()) == {
        "risk_score",
        "risk_level",
        "decision",
        "risk_factors",
    }


# --------------------------------------------------------------------------
# 1-3. Low / Medium / High scenarios
# --------------------------------------------------------------------------

def test_low_risk_scenario():
    # Confident genuine voice, matching speaker, no risky context.
    result = risk(synthetic_score=0.02, speaker_match=0.97)
    assert result.risk_score == pytest.approx(1.9)
    assert result.risk_level == "LOW"
    assert result.decision == "ALLOW"
    assert result.risk_factors == []


def test_medium_risk_scenario():
    result = risk(synthetic_score=0.5, speaker_match=0.5)
    assert result.risk_score == pytest.approx(40.0)
    assert result.risk_level == "MEDIUM"
    assert result.decision == "VERIFY"


def test_high_risk_scenario():
    result = risk(
        synthetic_score=0.97,
        speaker_match=0.03,
        context={"transfer": True, "otp": True, "urgent": True, "unknown_caller": True},
        transaction_amount=250_000,
    )
    assert result.risk_score == pytest.approx(97.6)
    assert result.risk_level == "HIGH"
    assert result.decision == "BLOCK"
    assert result.risk_score > 70


# --------------------------------------------------------------------------
# 4-8. Individual context signals each raise risk and add a factor
# --------------------------------------------------------------------------

BASE_KWARGS = dict(synthetic_score=0.2, speaker_match=0.8)  # -> risk_score 16.0


def test_baseline_without_context():
    assert risk(**BASE_KWARGS).risk_score == pytest.approx(16.0)


def test_transfer_context():
    result = risk(**BASE_KWARGS, context={"transfer": True})
    assert result.risk_score == pytest.approx(21.0)  # +0.25 weight / 1.0 * 0.20 * 100
    assert result.risk_score > 16.0
    assert "Money transfer requested" in result.risk_factors


def test_otp_context():
    result = risk(**BASE_KWARGS, context={"otp": True})
    assert result.risk_score == pytest.approx(21.0)
    assert "OTP request detected" in result.risk_factors


def test_urgent_context():
    result = risk(**BASE_KWARGS, context={"urgent": True})
    assert result.risk_score == pytest.approx(19.0)  # +0.15 weight
    assert result.risk_score > 16.0
    assert "Urgent request detected" in result.risk_factors


def test_unknown_caller_context():
    result = risk(**BASE_KWARGS, context={"unknown_caller": True})
    assert result.risk_score == pytest.approx(19.0)
    assert "Unknown caller" in result.risk_factors


def test_high_value_transaction():
    below = risk(**BASE_KWARGS, transaction_amount=HIGH_VALUE_THRESHOLD_INR - 1)
    at = risk(**BASE_KWARGS, transaction_amount=HIGH_VALUE_THRESHOLD_INR)

    assert "High-value transaction" not in below.risk_factors
    assert below.risk_score == pytest.approx(16.0)

    assert "High-value transaction" in at.risk_factors
    assert at.risk_score == pytest.approx(20.0)  # +0.20 weight
    assert at.risk_score > below.risk_score


def test_context_none_equals_no_signals():
    assert risk(**BASE_KWARGS, context=None).risk_score == risk(
        **BASE_KWARGS, context={}
    ).risk_score


def test_unknown_context_keys_are_ignored():
    result = risk(**BASE_KWARGS, context={"not_a_real_signal": True})
    assert result.risk_score == pytest.approx(16.0)


def test_all_context_signals_normalised_to_one():
    # transfer+otp+urgent+unknown_caller (0.25+0.25+0.15+0.15) plus high-value
    # (0.20) sum to exactly 1.0 -> context component maxes out at 0.20 * 100.
    result = risk(
        synthetic_score=0.0,
        speaker_match=1.0,
        context={"transfer": True, "otp": True, "urgent": True, "unknown_caller": True},
        transaction_amount=1_000_000,
    )
    assert result.risk_score == pytest.approx(20.0)


# --------------------------------------------------------------------------
# 9. Risk score clamping
# --------------------------------------------------------------------------

def test_risk_score_clamped_to_max_100():
    result = risk(
        synthetic_score=1.0,
        speaker_match=0.0,
        context={"transfer": True, "otp": True, "urgent": True, "unknown_caller": True},
        transaction_amount=10**9,
    )
    assert result.risk_score == 100.0
    assert result.risk_score <= 100.0
    assert result.risk_level == "HIGH"
    assert result.decision == "BLOCK"


def test_risk_score_clamped_to_min_0():
    result = risk(synthetic_score=0.0, speaker_match=1.0)
    assert result.risk_score == 0.0
    assert result.risk_score >= 0.0
    assert result.risk_level == "LOW"
    assert result.decision == "ALLOW"


@pytest.mark.parametrize(
    "syn, match, ctx, amt",
    [
        (0.0, 1.0, None, 0),
        (1.0, 0.0, {"transfer": True, "otp": True}, 500_000),
        (0.5, 0.5, {"urgent": True}, 100_000),
        (0.123, 0.456, {"unknown_caller": True}, 99_999),
    ],
)
def test_risk_score_always_within_bounds(syn, match, ctx, amt):
    result = calculate_risk(syn, match, ctx, amt)
    assert 0.0 <= result.risk_score <= 100.0


# --------------------------------------------------------------------------
# 10. Speaker mismatch calculation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "speaker_match, expected_risk",
    [
        (1.0, 0.0),    # perfect match -> mismatch 0
        (0.7, 9.0),    # mismatch 0.30 -> 0.30 * 0.30 * 100
        (0.25, 22.5),  # mismatch 0.75 -> 0.75 * 0.30 * 100
        (0.0, 30.0),   # total mismatch -> 1.0 * 0.30 * 100
    ],
)
def test_speaker_mismatch_drives_risk(speaker_match, expected_risk):
    result = risk(synthetic_score=0.0, speaker_match=speaker_match)
    assert result.risk_score == pytest.approx(expected_risk)


def test_speaker_mismatch_factor_only_when_significant():
    assert "Speaker identity mismatch" not in risk(speaker_match=0.9).risk_factors
    assert "Speaker identity mismatch" in risk(speaker_match=0.2).risk_factors


# --------------------------------------------------------------------------
# 11-12. Decision boundaries at risk 30 and 70
# --------------------------------------------------------------------------

def test_boundary_just_below_30_is_low():
    # mismatch 0.9991 -> 29.973 -> rounds to 29.97
    result = risk(synthetic_score=0.0, speaker_match=0.0009)
    assert result.risk_score < 30
    assert result.risk_level == "LOW"
    assert result.decision == "ALLOW"


def test_boundary_at_30_is_medium():
    result = risk(synthetic_score=0.0, speaker_match=0.0)  # exactly 30.0
    assert result.risk_score == 30.0
    assert result.risk_level == "MEDIUM"
    assert result.decision == "VERIFY"


def test_boundary_at_70_is_medium():
    result = risk(synthetic_score=0.8, speaker_match=0.0)  # 0.4 + 0.3 -> 70.0
    assert result.risk_score == 70.0
    assert result.risk_level == "MEDIUM"
    assert result.decision == "VERIFY"


def test_boundary_just_above_70_is_high():
    result = risk(synthetic_score=0.82, speaker_match=0.0)  # 0.41 + 0.3 -> 71.0
    assert result.risk_score > 70
    assert result.risk_level == "HIGH"
    assert result.decision == "BLOCK"


# --------------------------------------------------------------------------
# Risk factors wording
# --------------------------------------------------------------------------

def test_synthetic_factor_threshold():
    assert "High synthetic voice probability" not in risk(synthetic_score=0.49).risk_factors
    assert "High synthetic voice probability" in risk(synthetic_score=0.5).risk_factors


def test_factors_are_human_readable_strings():
    result = risk(
        synthetic_score=0.9,
        speaker_match=0.1,
        context={"transfer": True, "otp": True, "urgent": True, "unknown_caller": True},
        transaction_amount=200_000,
    )
    assert result.risk_factors == [
        "High synthetic voice probability",
        "Speaker identity mismatch",
        "Money transfer requested",
        "OTP request detected",
        "Urgent request detected",
        "Unknown caller",
        "High-value transaction",
    ]


# --------------------------------------------------------------------------
# 13-15. Invalid inputs raise ValueError
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.01, -0.5, 2.0, -0.0001])
def test_invalid_synthetic_score_raises(bad):
    with pytest.raises(ValueError):
        calculate_risk(bad, 0.5, None, 0.0)


@pytest.mark.parametrize("bad", [1.5, -0.01, 100.0, -1.0])
def test_invalid_speaker_match_raises(bad):
    with pytest.raises(ValueError):
        calculate_risk(0.5, bad, None, 0.0)


@pytest.mark.parametrize("bad", [-1, -0.01, -100000])
def test_negative_transaction_amount_raises(bad):
    with pytest.raises(ValueError):
        calculate_risk(0.5, 0.5, None, bad)


def test_valid_boundary_inputs_accepted():
    # 0 and 1 are inside the allowed range; amount 0 is allowed.
    for score in (0.0, 1.0):
        for match in (0.0, 1.0):
            calculate_risk(score, match, None, 0.0)
