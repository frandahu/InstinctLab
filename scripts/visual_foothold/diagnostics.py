"""Sample-aware summaries for recorded controller totals (no simulator imports)."""


def controller_stat_delta(current, previous, names):
    """Difference pre-reset totals, allowing signed-error sums to decrease."""
    delta = current - previous
    monotonic = [i for i, name in enumerate(names) if name not in (
        "signed_error_x_sum_m", "signed_error_y_sum_m",
    )]
    if (delta[:, monotonic] < -1e-5).any():
        raise RuntimeError("Controller totals reset without a recorded episode boundary")
    return delta


def summarize_totals(totals):
    """Return ratios with explicit numerator/denominator and JSON-null empty values."""
    definitions = {
        "valid_target_fraction": ("valid_ticks", "ticks"),
        "moving_target_fraction": ("valid_ticks", "moving_ticks"),
        "support_fraction": ("support_ticks", "ticks"),
        "target_generation_rate": ("targets_generated", "planning_attempts"),
        "target_retention_rate": ("target_retained_ticks", "target_exposure_ticks"),
        "corridor_observed_fraction": ("corridor_observed_samples", "corridor_required_samples"),
        "touchdown_error_mean_m": ("error_sum_m", "commanded_touchdowns"),
        "touchdown_hit_5cm": ("hits", "commanded_touchdowns"),
        "visual_risk_fraction": ("visual_risk", "risk_samples"),
        "payment_confirmation_ratio": ("reward_payments", "confirmed_commanded_touchdowns"),
        "touchdown_signed_error_x_m": ("signed_error_x_sum_m", "signed_error_samples"),
        "touchdown_signed_error_y_m": ("signed_error_y_sum_m", "signed_error_samples"),
        "touchdown_abs_error_x_m": ("abs_error_x_sum_m", "signed_error_samples"),
        "touchdown_abs_error_y_m": ("abs_error_y_sum_m", "signed_error_samples"),
        "touchdown_horizontal_error_m": ("error_xy_sum_m", "signed_error_samples"),
        "projection_changed_fraction": ("projection_changed", "corrections_requested"),
        "correction_erased_fraction": ("corrections_erased", "corrections_requested"),
        "correction_opposed_fraction": ("corrections_opposed", "corrections_requested"),
    }
    result = {}
    for name, (numerator, denominator) in definitions.items():
        samples = totals.get(denominator, 0)
        value = totals.get(numerator)
        result[name] = {"value": value / samples if samples > 0 and value is not None else None,
                        "numerator": value, "denominator": samples,
                        "numerator_name": numerator, "denominator_name": denominator}
    return result
