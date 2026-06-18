from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from quantcortex.backtest.conformance import (
    TARGET_TAPE_COLUMNS,
    TARGET_TAPE_SCHEMA_VERSION,
    target_tape_from_payload,
    target_tape_to_payload,
    target_tape_to_weights,
    validate_target_tape,
    weights_to_target_tape,
)
from quantcortex.backtest.costs.transaction_costs import TransactionCostModel
from quantcortex.backtest.engines.event_driven import EventDrivenBacktest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "conformance"


def _fixture_tape() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_ROOT / "target_tape.csv")


def _target_tape_schema_validator() -> Draft202012Validator:
    schema = json.loads(
        (REPO_ROOT / "schemas" / "canonical_target_tape.schema.json").read_text()
    )
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _wire_case_payload(case: str) -> dict[str, object]:
    payload = copy.deepcopy(
        target_tape_to_payload(_fixture_tape(), expected_symbols=["A", "B"])
    )
    records = payload["records"]
    assert isinstance(records, list)

    if case == "valid":
        return payload
    if case == "integral_float_schema_version":
        payload["schema_version"] = 1.0
    elif case == "boolean_schema_version":
        payload["schema_version"] = True
    elif case == "nonintegral_schema_version":
        payload["schema_version"] = 1.5
    elif case == "unsupported_schema_version":
        payload["schema_version"] = 2
    elif case == "string_max_gross":
        payload["max_gross"] = "1.0"
    elif case == "negative_target_weight":
        records[0]["target_weight"] = -5e-13
    elif case == "oversized_binary64_number":
        oversized = 10**1000
        payload["max_gross"] = oversized
        records[0]["target_weight"] = oversized
    elif case == "maximum_binary64_number":
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        payload["max_gross"] = maximum
        records[0]["target_weight"] = maximum
    elif case in {
        "date_only_timestamp",
        "lowercase_timestamp",
        "year_zero_timestamp",
        "invalid_calendar_date",
        "fractional_timestamp",
        "maximum_offset_timestamp",
        "local_hour_24",
        "local_minute_60",
        "local_second_60",
        "offset_hour_24",
        "offset_minute_60",
    }:
        replacements = {
            "date_only_timestamp": "2024-01-01",
            "lowercase_timestamp": "2024-01-01t00:00:00z",
            "year_zero_timestamp": "0000-01-01T00:00:00Z",
            "invalid_calendar_date": "2023-02-29T00:00:00Z",
            "fractional_timestamp": "2024-01-01T00:00:00.123456789Z",
            "maximum_offset_timestamp": "2024-01-01T00:00:00+23:59",
            "local_hour_24": "2024-01-01T24:00:00Z",
            "local_minute_60": "2024-01-01T00:60:00Z",
            "local_second_60": "2024-01-01T00:00:60Z",
            "offset_hour_24": "2024-01-01T00:00:00+24:00",
            "offset_minute_60": "2024-01-01T00:00:00+00:60",
        }
        original = records[0]["decision_timestamp"]
        for record in records:
            if record["decision_timestamp"] == original:
                record["decision_timestamp"] = replacements[case]
    elif case in {"blank_symbol", "outer_whitespace_symbol"}:
        replacement = " " if case == "blank_symbol" else " A"
        symbols = payload["symbols"]
        assert isinstance(symbols, list)
        symbols[0] = replacement
        for record in records:
            if record["symbol"] == "A":
                record["symbol"] = replacement
    elif case == "unknown_payload_field":
        payload["unexpected"] = True
    elif case == "unknown_record_field":
        records[0]["unexpected"] = True
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(f"unknown wire test case: {case}")
    return payload


def test_canonical_target_tape_round_trips():
    tape = validate_target_tape(_fixture_tape(), expected_symbols=["A", "B"])
    weights = target_tape_to_weights(tape, expected_symbols=["A", "B"])
    restored = weights_to_target_tape(weights)

    pd.testing.assert_frame_equal(restored, tape)
    assert tuple(restored.columns) == TARGET_TAPE_COLUMNS


def test_canonical_target_tape_json_payload_round_trips():
    tape = validate_target_tape(_fixture_tape(), expected_symbols=["A", "B"])
    payload = target_tape_to_payload(
        tape,
        max_gross=1.0,
        expected_symbols=["A", "B"],
    )
    restored = target_tape_from_payload(payload)

    assert payload["schema_version"] == TARGET_TAPE_SCHEMA_VERSION
    assert payload["symbols"] == ["A", "B"]
    assert payload["max_gross"] == 1.0
    assert payload["records"][0]["decision_timestamp"].endswith("Z")
    pd.testing.assert_frame_equal(restored, tape)


@pytest.mark.parametrize(
    ("case", "accepted"),
    [
        ("valid", True),
        ("integral_float_schema_version", True),
        ("boolean_schema_version", False),
        ("nonintegral_schema_version", False),
        ("unsupported_schema_version", False),
        ("string_max_gross", False),
        ("negative_target_weight", False),
        ("oversized_binary64_number", False),
        ("maximum_binary64_number", True),
        ("date_only_timestamp", False),
        ("lowercase_timestamp", False),
        ("year_zero_timestamp", False),
        ("invalid_calendar_date", False),
        ("fractional_timestamp", True),
        ("maximum_offset_timestamp", True),
        ("local_hour_24", False),
        ("local_minute_60", False),
        ("local_second_60", False),
        ("offset_hour_24", False),
        ("offset_minute_60", False),
        ("blank_symbol", False),
        ("outer_whitespace_symbol", False),
        ("unknown_payload_field", False),
        ("unknown_record_field", False),
    ],
)
def test_target_tape_schema_and_runtime_align_on_wire_constraints(case, accepted):
    """Keep primitive wire validation aligned without duplicating semantics."""
    payload = _wire_case_payload(case)
    schema_accepts = not list(_target_tape_schema_validator().iter_errors(payload))
    try:
        target_tape_from_payload(payload)
    except (TypeError, ValueError):
        runtime_accepts = False
    else:
        runtime_accepts = True

    assert schema_accepts is accepted
    assert runtime_accepts is accepted


def test_canonical_target_tape_payload_survives_strict_json_round_trip():
    tape = validate_target_tape(_fixture_tape(), expected_symbols=["A", "B"])
    payload = target_tape_to_payload(tape, expected_symbols=["A", "B"])
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)

    _target_tape_schema_validator().validate(decoded)
    restored = target_tape_from_payload(decoded)

    pd.testing.assert_frame_equal(restored, tape)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_target_tape_payload_rejects_nonfinite_json_numbers(value):
    payload = target_tape_to_payload(_fixture_tape(), expected_symbols=["A", "B"])
    payload["records"][0]["target_weight"] = value

    with pytest.raises(ValueError, match="must be finite binary64"):
        target_tape_from_payload(payload)
    with pytest.raises(ValueError, match="Out of range float values"):
        json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(index=1), "incomplete symbols"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (
            lambda frame: frame.assign(
                target_weight=[0.8, 0.4, 0.0, 1.0]
            ),
            "gross exposure",
        ),
        (
            lambda frame: frame.assign(
                target_weight=[-0.1, 1.0, 0.0, 1.0]
            ),
            "long-only",
        ),
    ],
)
def test_canonical_target_tape_rejects_contract_violations(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_target_tape(mutator(_fixture_tape()), expected_symbols=["A", "B"])


def test_canonical_target_tape_rejects_symbols_that_collide_after_stripping():
    with pytest.raises(ValueError, match="unique after normalization"):
        validate_target_tape(_fixture_tape(), expected_symbols=["A", " A"])


def test_canonical_target_tape_payload_rejects_unknown_fields():
    payload = target_tape_to_payload(_fixture_tape(), expected_symbols=["A", "B"])
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="payload keys must be exactly"):
        target_tape_from_payload(payload)


def test_target_tape_to_weights_preserves_declared_symbol_order():
    weights = target_tape_to_weights(
        _fixture_tape(),
        expected_symbols=["B", "A"],
    )

    assert list(weights.columns) == ["B", "A"]


def test_target_tape_long_only_tolerance_preserves_positive_weights_only():
    tape = _fixture_tape()
    tape.loc[0, "target_weight"] = 5e-16
    tape.loc[1, "target_weight"] = -5e-13

    normalized = validate_target_tape(tape, expected_symbols=["A", "B"])

    assert normalized.loc[0, "target_weight"] == 5e-16
    assert normalized.loc[1, "target_weight"] == 0.0
    payload = target_tape_to_payload(normalized, expected_symbols=["A", "B"])
    roundtrip = target_tape_to_weights(
        target_tape_from_payload(payload),
        expected_symbols=["A", "B"],
    )
    assert roundtrip.iloc[0].to_dict() == {"A": 5e-16, "B": 0.0}

    tape.loc[1, "target_weight"] = -2e-12
    with pytest.raises(ValueError, match="long-only"):
        validate_target_tape(tape, expected_symbols=["A", "B"])


@pytest.mark.parametrize(
    ("path", "value", "error", "message"),
    [
        (("schema_version",), True, TypeError, "schema_version must be a JSON number"),
        (("max_gross",), "1.0", TypeError, "max_gross must be a JSON number"),
        (
            ("records", 0, "decision_timestamp"),
            "2024-01-01",
            ValueError,
            "must be an RFC 3339 date-time",
        ),
        (
            ("records", 0, "decision_timestamp"),
            1_704_067_200,
            TypeError,
            "must be an RFC 3339 string",
        ),
        (
            ("records", 0, "target_weight"),
            "1.0",
            TypeError,
            "target_weight must be a JSON number",
        ),
        (
            ("records", 0, "target_weight"),
            np.float64(1.0),
            TypeError,
            "target_weight must be a JSON number",
        ),
    ],
)
def test_canonical_target_tape_payload_enforces_published_json_types(
    path, value, error, message
):
    payload = target_tape_to_payload(_fixture_tape(), expected_symbols=["A", "B"])
    mutated = copy.deepcopy(payload)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(error, match=message):
        target_tape_from_payload(mutated)


def test_conformance_fixture_has_hand_computable_next_bar_returns():
    weights = target_tape_to_weights(
        _fixture_tape(),
        expected_symbols=["A", "B"],
    )
    prices = pd.read_csv(FIXTURE_ROOT / "prices.csv", index_col="timestamp")
    prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)
    result = EventDrivenBacktest(
        TransactionCostModel(commission=0.0, slippage=0.0),
        capital=1.0,
    ).run(weights, prices, cash_returns=pd.Series(0.0, index=prices.index))

    expected = pd.Series(
        [0.0, 0.0, 0.10, 0.0, 0.20],
        index=prices.index,
    )
    pd.testing.assert_series_equal(result.returns, expected)
    assert result.metadata["execution_timing"] == "next_bar_close"


def test_committed_contract_schemas_are_versioned_and_specific():
    target_schema = _target_tape_schema_validator().schema
    contract_schema = json.loads(
        (REPO_ROOT / "schemas" / "evaluation_contract.schema.json").read_text()
    )

    assert target_schema["properties"]["schema_version"]["const"] == (
        TARGET_TAPE_SCHEMA_VERSION
    )
    assert target_schema["required"] == [
        "schema_version",
        "symbols",
        "max_gross",
        "records",
    ]
    record = target_schema["properties"]["records"]["items"]
    assert record["required"] == list(TARGET_TAPE_COLUMNS)
    assert contract_schema["properties"]["schema_version"]["const"] == 1
    required_comparators = contract_schema["properties"]["comparators"]["required"]
    assert required_comparators == [
        "realized_exposure_attribution_control",
        "target_exposure_costed_comparator",
    ]
    assert contract_schema["properties"]["target_tape"][
        "additionalProperties"
    ] is False

    _target_tape_schema_validator().validate(
        target_tape_to_payload(_fixture_tape(), expected_symbols=["A", "B"])
    )
    contract_validator = Draft202012Validator(contract_schema)
    contract_validator.validate(
        json.loads(
            (REPO_ROOT / "paper" / "results" / "evaluation_contract.json").read_text()
        )
    )
