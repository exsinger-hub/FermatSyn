import argparse
from pathlib import Path

import pytest

from util.a2_eval_schedule import (
    add_evaluation_record_fields,
    positive_int,
    should_evaluate_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


def test_eval_schedule_uses_cadence_and_always_evaluates_final_epoch():
    assert all(should_evaluate_epoch(epoch, 10, 1) for epoch in range(1, 11))
    assert should_evaluate_epoch(1, 10, 3) is False
    assert should_evaluate_epoch(3, 10, 3) is True
    assert should_evaluate_epoch(9, 10, 3) is True
    assert should_evaluate_epoch(10, 10, 3) is True
    assert should_evaluate_epoch(10, 10, 20) is True


@pytest.mark.parametrize("value", ("0", "-2", "not-an-integer"))
def test_positive_int_rejects_invalid_eval_every(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(value)


def test_positive_int_and_schedule_accept_valid_values():
    assert positive_int("4") == 4
    with pytest.raises(ValueError, match="eval_every must be positive"):
        should_evaluate_epoch(1, 10, 0)
    with pytest.raises(ValueError, match="epoch must lie"):
        should_evaluate_epoch(11, 10, 1)


def test_skipped_epoch_record_has_null_guard_and_no_validation_fields():
    record = add_evaluation_record_fields(
        {"epoch": 1, "train_l1": 0.04},
        evaluation_performed=False,
    )
    assert record == {
        "epoch": 1,
        "train_l1": 0.04,
        "evaluation_performed": False,
        "axial_guard_pass": None,
    }
    assert "a2_l1" not in record


def test_epoch_record_accepts_validation_only_when_evaluation_was_performed():
    record = add_evaluation_record_fields(
        {"epoch": 3, "train_l1": 0.04},
        evaluation_performed=True,
        axial_guard_pass=False,
        val_metrics={"a2_l1": 0.03},
    )
    assert record["evaluation_performed"] is True
    assert record["axial_guard_pass"] is False
    assert record["a2_l1"] == 0.03

    with pytest.raises(ValueError, match="skipped epochs"):
        add_evaluation_record_fields(
            {"epoch": 1},
            evaluation_performed=False,
            val_metrics={"a2_l1": 0.03},
        )


def test_server26_launcher_forwards_eval_every_environment_setting():
    script = (ROOT / "scripts" / "run_a2_v2_arm_26.sh").read_text(
        encoding="utf-8"
    )
    assert "EVAL_EVERY=${A2V2_EVAL_EVERY:-1}" in script
    assert '--eval-every "${EVAL_EVERY}"' in script
