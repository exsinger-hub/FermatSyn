"""Validation scheduling helpers for A2-v2 training."""

import argparse


def positive_int(value):
    """Argparse type for strictly positive integer options."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def should_evaluate_epoch(epoch, total_epochs, eval_every):
    """Evaluate on the requested cadence and always on the final epoch."""

    epoch = int(epoch)
    total_epochs = int(total_epochs)
    eval_every = int(eval_every)
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive")
    if epoch < 1 or epoch > total_epochs:
        raise ValueError("epoch must lie in [1, total_epochs]")
    if eval_every < 1:
        raise ValueError("eval_every must be positive")
    return epoch == total_epochs or epoch % eval_every == 0


def add_evaluation_record_fields(
    record, evaluation_performed, axial_guard_pass=None, val_metrics=None
):
    """Attach truthful evaluation state without inventing skipped val fields."""

    result = dict(record)
    val_metrics = dict(val_metrics or {})
    evaluation_performed = bool(evaluation_performed)
    if evaluation_performed:
        if not isinstance(axial_guard_pass, bool):
            raise ValueError("evaluated epochs require a boolean axial guard")
        if not val_metrics:
            raise ValueError("evaluated epochs require validation metrics")
    elif axial_guard_pass is not None or val_metrics:
        raise ValueError("skipped epochs cannot contain validation results")
    result["evaluation_performed"] = evaluation_performed
    result["axial_guard_pass"] = (
        axial_guard_pass if evaluation_performed else None
    )
    if evaluation_performed:
        result.update(val_metrics)
    return result
