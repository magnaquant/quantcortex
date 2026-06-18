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
        (("schema_version",), True, TypeError, "schema_version must be an integer"),
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
    target_schema = json.loads(
        (REPO_ROOT / "schemas" / "canonical_target_tape.schema.json").read_text()
    )
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

    target_validator = Draft202012Validator(
        target_schema,
        format_checker=FormatChecker(),
    )
    target_validator.validate(
        target_tape_to_payload(_fixture_tape(), expected_symbols=["A", "B"])
    )
    contract_validator = Draft202012Validator(contract_schema)
    contract_validator.validate(
        json.loads(
            (REPO_ROOT / "paper" / "results" / "evaluation_contract.json").read_text()
        )
    )
