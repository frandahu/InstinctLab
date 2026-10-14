"""CPU contracts for frozen-policy reference perturbations and causal EWMA feedback."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from analyze_reference import compare_runs, response_contrast
from reference_diagnostics import ReferenceRecorder, summarize_reference
from test_core import ControllerFixture, flat_cloud, load, perception

reference = load("reference")


class ReferenceContracts(unittest.TestCase):
    def test_modes_change_actor_reference_and_compensate_only_after_past_touchdown(self):
        baseline = ControllerFixture(1, reference.ReferenceExperimentSettings(mode="baseline"))
        perturbed = ControllerFixture(1, reference.ReferenceExperimentSettings(mode="perturb", perturb_m=-.04))
        # Use actual defaults: training feedback must activate after the first
        # valid touchdown, rather than silently behaving as a baseline.
        compensated = ControllerFixture(1)
        self.assertEqual(compensated.core.reference_experiment.cfg.mode, "compensate")
        self.assertEqual(compensated.core.reference_experiment.cfg.min_samples, 1)
        for fixture in (baseline, perturbed, compensated):
            fixture.begin_swing()
        torch.testing.assert_close(perturbed.core.observation[:, 22:25] - baseline.core.observation[:, 22:25],
                                   torch.tensor([[-.04, 0., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(compensated.core.observation, baseline.core.observation)
        compensated.p[:, 0] = compensated.core.target + compensated.p[:, 1] + torch.tensor([.03, 0., 0.])
        compensated.force[:, 0] = 100
        compensated.step(8)  # +3 cm first-contact error, before confirmation.
        compensated.step(9)
        experiment = compensated.core.reference_experiment
        for tick in range(10, 41):
            compensated.step(tick)
        self.assertEqual(experiment.plan["foot"].item(), 0)
        self.assertEqual(experiment.plan["bias_samples"].item(), 1)
        torch.testing.assert_close(experiment.plan["requested"][:, :2] - experiment.plan["planned"][:, :2],
                                   torch.tensor([[-.03, 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(experiment.plan["applied"][:, :2] - experiment.plan["planned"][:, :2],
                                   torch.tensor([[-.04, 0.]]), atol=1e-6, rtol=0)
        self.assertFalse(experiment.plan["correction_erased"].item())

    def test_baseline_is_unchanged_and_small_perturbations_report_grid_erasure(self):
        # Independent flat geometry: 2 cm proposals tie the 4 cm lattice and
        # retain the plan, whereas 4 cm proposals reach a distinct safe command.
        offsets = torch.tensor([[.02, 0.], [-.02, 0.], [.04, 0.], [-.04, 0.],
                                [0., .02], [0., -.02], [0., .04], [0., -.04], [0., 0.]])
        n = len(offsets)
        cloud = flat_cloud(n)
        grid = perception.SupportGrid(perception.VisualFootholdSettings(), "cpu")
        inputs = (cloud, torch.ones(cloud.shape[:2], dtype=torch.bool), torch.zeros(n, 3),
                  torch.zeros(n, dtype=torch.long), torch.tensor([[1., 0.]]).expand(n, -1),
                  torch.ones(n)*.5, torch.zeros(n, 3))
        baseline, _, _ = grid.choose(*inputs)
        applied, valid, _ = grid.choose(*inputs, offset_xy=offsets)
        self.assertTrue(valid.all())
        torch.testing.assert_close(grid.reference["planned"], baseline)
        torch.testing.assert_close(applied[[0, 1, 4, 5, 8]], baseline[[0, 1, 4, 5, 8]])
        torch.testing.assert_close(applied[[2, 3, 6, 7], :2] - baseline[[2, 3, 6, 7], :2], offsets[[2, 3, 6, 7]],
                                   atol=1e-6, rtol=0)
        experiment = reference.ReferenceExperiment(n, "cpu")
        ids = torch.arange(n)
        experiment.commit(ids, torch.zeros(n, dtype=torch.long), grid.reference, torch.zeros(n),
                          torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long))
        self.assertEqual(experiment.plan["correction_erased"].tolist(), [True, True, False, False, True, True, False, False, False])

    def test_projection_preserves_reach_footprint_and_unknown_corridor_rules(self):
        cfg = perception.VisualFootholdSettings(max_step=.32)
        grid = perception.SupportGrid(cfg, "cpu")
        cloud = flat_cloud(3)
        visible = torch.ones(cloud.shape[:2], dtype=torch.bool)
        visible[1] = cloud[1, :, 0] >= .32  # Safe far landings, blind swing corridor.
        visible[2] = False
        applied, valid, _ = grid.choose(cloud, visible, torch.zeros(3, 3), torch.zeros(3, dtype=torch.long),
                                       torch.tensor([[1., 0.]]).expand(3, -1), torch.ones(3)*.5, torch.zeros(3, 3),
                                       offset_xy=torch.tensor([[.04, 0.]]).expand(3, -1))
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertLessEqual(applied[0, :2].norm().item(), .32 + 1e-6)
        _, _, safe, _, _ = grid.build(cloud, visible)
        self.assertEqual(grid.sample(safe[:1], applied[:1, None, :2]).item(), 1)
        self.assertGreater((grid.reference["requested"][0, :2]).norm().item(), .32)

    def test_first_contact_signed_error_is_frozen_confirmed_once_and_reset_isolated(self):
        fixture = ControllerFixture(2, reference.ReferenceExperimentSettings(mode="compensate", min_samples=1))
        fixture.begin_swing()
        core = fixture.core
        history = core.reference_experiment
        self.assertTrue((history.samples == 0).all())
        torch.testing.assert_close(history.plan["planned"], history.plan["applied"])
        fixture.p[:, 0] = core.target + fixture.p[:, 1] + torch.tensor([.03, -.02, .01])
        fixture.force[:, 0] = 100
        fixture.step(8)
        self.assertTrue((history.samples == 0).all())  # An unconfirmed candidate cannot train.
        latched = history.candidate["actual"].clone()
        fixture.p[:, 0, 0] += .025  # Unreliable transfer at confirmation; retain the valid old-frame measurement.
        fixture.step(9)
        torch.testing.assert_close(history.candidate["actual"], latched)
        torch.testing.assert_close(history.candidate["error"][:, 0], torch.tensor([[.03, -.02, .01]]).expand(2, -1))
        torch.testing.assert_close(history.bias[:, 0], torch.tensor([[.03, -.02]]).expand(2, -1))
        self.assertEqual(history.samples.tolist(), [[1, 0], [1, 0]])
        fixture.step(10)
        self.assertEqual(history.samples.tolist(), [[1, 0], [1, 0]])
        snapshot = history.snapshot()
        core.reset(torch.tensor([0]))
        self.assertEqual(history.samples.tolist(), [[0, 0], [1, 0]])
        self.assertEqual(snapshot["bias_samples"].tolist(), [[1, 0], [1, 0]])

    def test_ewma_sign_foot_separation_clamp_and_heading_rotation(self):
        experiment = reference.ReferenceExperiment(2, "cpu", reference.ReferenceExperimentSettings(
            mode="compensate", ewma_alpha=.25, min_samples=2,
        ))
        confirmed = torch.tensor([[True, False], [False, True]])
        aborted = torch.zeros_like(confirmed)
        experiment.candidate["error_heading"][:] = torch.tensor([.04, -.02])
        experiment.finish(confirmed, aborted, torch.zeros(2))
        self.assertTrue((experiment.propose(torch.arange(2), torch.tensor([0, 1]), torch.eye(3).expand(2, -1, -1)) == 0).all())
        experiment.candidate["error_heading"][0, 0] = torch.tensor([.02, .02])
        experiment.candidate["error_heading"][1, 1] = torch.tensor([.4, 0.])
        experiment.finish(confirmed, aborted, torch.ones(2))
        torch.testing.assert_close(experiment.bias[0, 0], torch.tensor([.035, -.01]))
        self.assertEqual(experiment.samples.tolist(), [[2, 0], [0, 2]])
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        offsets = experiment.propose(torch.arange(2), torch.tensor([0, 1]), rotation.expand(2, -1, -1))
        torch.testing.assert_close(offsets[0], torch.tensor([-.01, -.035]))
        self.assertAlmostEqual(offsets[1].norm().item(), .04, places=6)
        # Positive heading X error produces negative heading X correction.
        self.assertLess(offsets[1, 1].item(), 0)

    def test_aborted_and_outlier_contacts_do_not_train_bias(self):
        fixture = ControllerFixture(2)
        fixture.begin_swing()
        fixture.candidate_contact(8)
        fixture.force[0, 0] = 0
        fixture.core.reference_experiment.cfg.max_update_error_m = .01
        fixture.step(9)
        experiment = fixture.core.reference_experiment
        self.assertEqual(experiment.event_status.tolist(), [[2, 0], [1, 0]])
        self.assertTrue((experiment.samples == 0).all())
        self.assertEqual(experiment.totals["ewma_rejected"].tolist(), [0., 1.])
        self.assertEqual(experiment.totals["signed_error_samples"].tolist(), [0., 1.])

    def test_csv_uses_terminal_snapshot_and_excludes_unconfirmed_errors(self):
        fixture = ControllerFixture(3, reference.ReferenceExperimentSettings(mode="perturb", perturb_m=.04))
        snapshots = []
        for tick in range(6):
            fixture.step(tick)
            snapshots.append(fixture.core.reference_experiment.snapshot())
        fixture.force[:, 0] = 0
        fixture.step(6)
        fixture.step(7)
        fixture.candidate_contact(8)
        pending = fixture.core.reference_experiment.snapshot()
        pending["pending"] = fixture.core.candidate_valid.clone()
        fixture.core.reset(torch.tensor([1]))
        fixture.force[2, 0] = 0
        fixture.step(9)
        terminal = fixture.core.reference_experiment.snapshot()
        terminal["pending"] = fixture.core.candidate_valid.clone()
        fixture.core.reset(torch.arange(3))
        with tempfile.TemporaryDirectory() as directory:
            with ReferenceRecorder(directory) as recorder:
                for tick, snapshot in enumerate(snapshots):
                    snapshot["pending"] = torch.zeros(3, 2, dtype=torch.bool)
                    recorder.record(tick, [0, 0, 0], snapshot, [False, False, False])
                recorder.record(8, [0, 0, 0], pending, [False, True, False])
                recorder.record(9, [0, 1, 0], terminal, [True, False, False])
            summary = recorder.summary()
            self.assertEqual(summary["reference_commands"], 3)
            self.assertEqual(summary["confirmed"]["samples"], 1)
            self.assertEqual(summary["reset_unconfirmed_candidates"], 1)
            self.assertEqual(summary["aborted_candidates"], 1)
            self.assertAlmostEqual(summary["confirmed"]["signed_error_xy_mean_m"][0], .02, places=6)
            self.assertIsNone(summary["per_foot"]["right"]["horizontal_error_mean_m"])
            with (Path(directory) / "touchdown_events.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            actual = next(row for row in rows if row["status"] == "confirmed")
            self.assertAlmostEqual(float(actual["actual_x_a_m"]) - float(actual["applied_x_a_m"]), .02, places=6)
            self.assertEqual(actual["frame"], "first_contact_anchor")
            # Analyze exactly the files that a remote rollout produces.
            (Path(directory) / "summary.json").write_text(json.dumps({
                "status": "complete", "comparison_protocol_sha256": "matched-protocol",
                "reference_experiment": {"mode": "baseline"}, "termination_counts": {}, "valid_target_fraction": .5,
            }), encoding="utf-8")
            report = compare_runs([directory])
            self.assertEqual(report["runs"][0]["reference_metrics"], summary)
            other = Path(directory) / "different_protocol"
            other.mkdir()
            for name in ("reference_plans.csv", "touchdown_events.csv"):
                (other / name).write_bytes((Path(directory) / name).read_bytes())
            metadata = json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))
            metadata["comparison_protocol_sha256"] = "different-checkpoint"
            (other / "summary.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical checkpoint"):
                compare_runs([directory, other])

    def test_response_contrast_removes_constant_bias_and_absent_samples_stay_null(self):
        rows = []
        for applied in (-.04, .04):
            rows.append(dict(status="confirmed", env_id=0, episode=0, slot=0, foot=0,
                             planned_x_a_m=.24, applied_x_a_m=.24+applied, actual_x_a_m=.24+applied+.03))
        contrast = response_contrast(rows[:1], rows[1:], "x")
        self.assertEqual(contrast["matched_confirmed_slots"], 1)
        self.assertAlmostEqual(contrast["response_secant_proxy"], 1.)
        self.assertIsNone(response_contrast([], rows, "x")["response_secant_proxy"])
        empty = summarize_reference([], [])
        self.assertEqual(empty["confirmed"]["samples"], 0)
        self.assertIsNone(empty["confirmed"]["horizontal_error_mean_m"])
        self.assertIsNone(empty["correction_erased_fraction"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
