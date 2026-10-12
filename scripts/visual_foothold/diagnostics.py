"""Sample-aware summaries for recorded controller totals (no simulator imports)."""


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
    }
    result = {}
    for name, (numerator, denominator) in definitions.items():
        samples = totals.get(denominator, 0)
        value = totals.get(numerator)
        result[name] = {"value": value / samples if samples > 0 and value is not None else None,
                        "numerator": value, "denominator": samples,
                        "numerator_name": numerator, "denominator_name": denominator}
    return result
