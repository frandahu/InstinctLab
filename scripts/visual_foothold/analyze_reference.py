"""Compare frozen-policy reference experiments from recorded CSVs; CPU/std-lib only."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from reference_diagnostics import summarize_reference


def load_run(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete":
        raise ValueError(f"Incomplete/error playback cannot enter a comparison: {directory}")
    with (directory / "reference_plans.csv").open(newline="", encoding="utf-8") as stream:
        plans = list(csv.DictReader(stream))
    with (directory / "touchdown_events.csv").open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    return summary, plans, events


def response_contrast(left, right, axis):
    """Pair observed slots; ambiguous/missing contacts are excluded and counted.

    Same seeds/slots do NOT imply equal closed-loop states after interventions.
    This descriptive secant is explicitly not a causal response identification.
    """
    def index(rows):
        result = {}
        for row in rows:
            if row["status"] == "confirmed":
                key = tuple(int(row[k]) for k in ("env_id", "episode", "slot", "foot"))
                result.setdefault(key, []).append(row)
        return result

    a, b = index(left), index(right)
    pairs = [(a[k][0], b[k][0]) for k in a.keys() & b.keys() if len(a[k]) == len(b[k]) == 1]
    def delta(row, name):
        return float(row[f"{name}_{axis}_a_m"]) - float(row[f"planned_{axis}_a_m"])
    applied = [delta(y, "applied") - delta(x, "applied") for x, y in pairs]
    landing = [delta(y, "actual") - delta(x, "actual") for x, y in pairs]
    mean_applied = statistics.mean(applied) if applied else None
    mean_landing = statistics.mean(landing) if landing else None
    return {
        "axis": axis, "matched_confirmed_slots": len(pairs),
        "left_confirmed_slots": len(a), "right_confirmed_slots": len(b),
        "applied_offset_contrast_m": mean_applied, "landing_offset_contrast_m": mean_landing,
        "response_secant_proxy": mean_landing / mean_applied if mean_applied is not None and abs(mean_applied) > 0.001 else None,
        "interpretation": "descriptive matched-slot contrast; closed-loop states and surviving contacts may differ; "
                          "null when projection erases the contrast or contacts are absent",
    }


def compare_runs(directories):
    loaded = [(str(Path(path)), *load_run(path)) for path in directories]
    protocols = {summary.get("comparison_protocol_sha256") for _, summary, _, _ in loaded}
    if None in protocols or len(protocols) != 1:
        raise ValueError("Compare only identical checkpoint/config/source/seed/env-count/step-budget protocols")
    results = [{"directory": path, "experiment": summary["reference_experiment"],
                "reference_metrics": summarize_reference(plans, events),
                "termination_counts": summary["termination_counts"],
                "valid_target_fraction": summary["valid_target_fraction"]}
               for path, summary, plans, events in loaded]
    contrasts = []
    for i, (path_a, summary_a, _, events_a) in enumerate(loaded):
        for path_b, summary_b, _, events_b in loaded[i+1:]:
            a, b = summary_a["reference_experiment"], summary_b["reference_experiment"]
            if a["mode"] == b["mode"] == "perturb" and a["perturb_axis"] == b["perturb_axis"]:
                if abs(a["perturb_m"] + b["perturb_m"]) < 1e-8:
                    contrasts.append(dict(left=path_a, right=path_b,
                                          **response_contrast(events_a, events_b, a["perturb_axis"])))
    return {"comparison_protocol_sha256": next(iter(protocols)), "runs": results, "perturbation_contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--output", required=True, help="New comparison JSON path; existing files are never overwritten")
    args = parser.parse_args()
    result = compare_runs(args.directories)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    for run in result["runs"]:
        metrics = run["reference_metrics"]
        print(f"{run['directory']}: mode={run['experiment']['mode']} "
              f"commands={metrics['reference_commands']} confirmed={metrics['confirmed']['samples']} "
              f"xy_error={metrics['confirmed']['horizontal_error_mean_m']} "
              f"erased={metrics['corrections_erased']}/{metrics['corrections_requested']}")
    print(f"Comparison: {output}")


if __name__ == "__main__":
    main()
