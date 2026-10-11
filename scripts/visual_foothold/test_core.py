"""CPU contracts for geometry, measured-state FK, and command lifecycle.

Run with a Torch interpreter; no Isaac Sim app is launched. Package loading here
avoids InstinctLab's simulator imports, not the production algorithms.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/instinctlab/instinctlab/tasks/parkour/config/g1/visual_foothold"
PACKAGE = "visual_foothold_cpu"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def load(name, directory=SOURCE, package_name=PACKAGE):
    spec = importlib.util.spec_from_file_location(package_name + "." + name, directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kinematics = load("kinematics")
perception = load("perception")
controller = load("controller")
mdp = load("mdp")


def flat_cloud(n=1):
    # Independent synthetic visible floor, with more samples than the grid.
    y, x = torch.meshgrid(torch.arange(-0.65, 0.66, 0.02), torch.arange(-0.35, 1.16, 0.02), indexing="ij")
    p = torch.stack((x.flatten(), y.flatten(), torch.zeros(x.numel())), -1)
    return p[None].expand(n, -1, -1).clone()


class VisualFootholdContracts(unittest.TestCase):
    def test_adapter_cache_and_pre_reset_snapshot_are_isolated(self):
        from types import SimpleNamespace as NS
        urdf = SOURCE.parents[2] / "urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
        names = [j.get("name") for j in ET.parse(urdf).getroot().findall("joint") if j.get("type") != "fixed"]
        settings = perception.VisualFootholdSettings(joint_position_noise=0., joint_velocity_noise=0., depth_noise_std=0.)
        data = NS(joint_pos=torch.zeros(2, len(names)), joint_vel=torch.zeros(2, len(names)),
                  projected_gravity_b=torch.tensor([[0., 0., -1.]]).expand(2, -1))
        camera = NS(cfg=NS(offset=NS(convention="world", rot=(1., 0., 0., 0.), pos=(0., 0., 0.)),
                           pattern_cfg=NS(height=36, width=64), update_period=0.02),
                    data=NS(output={"distance_to_image_plane": torch.ones(2, 36, 64, 1)}),
                    ray_directions=torch.tensor([1., 0., 0.]).expand(2, 36*64, 3),
                    ray_starts=torch.zeros(2, 36*64, 3))
        contacts = NS(data=NS(net_forces_w=torch.tensor([0., 0., 100.]).expand(2, 2, 3)),
                      find_bodies=lambda name: ([0 if name.startswith("left") else 1], [name]))
        # No terrain importer, robot world pose, or hit points are supplied.
        env = NS(cfg=NS(visual_foothold=settings), num_envs=2, device="cpu", step_dt=0.02,
                 scene={"robot": NS(data=data, joint_names=names), "camera": camera, "contact_forces": contacts},
                 command_manager=NS(get_command=lambda name: torch.zeros(2, 3)), common_step_counter=0,
                 visual_recording=True)
        adapter = mdp.state(env)
        adapter.update()
        env.common_step_counter = 1
        mdp.capture_before_reset(env)
        before = adapter.core.observation[1].clone()
        for _ in range(3):
            mdp.visual_observation(env)
            mdp.visual_reward(env, "swing")
        torch.testing.assert_close(adapter.core.time, torch.ones(2)*0.02)
        adapter.reset(torch.tensor([0]))
        adapter.update()
        torch.testing.assert_close(adapter.core.time, torch.tensor([0., 0.02]))
        torch.testing.assert_close(adapter.core.observation[1], before)
        torch.testing.assert_close(env.visual_snapshot["time"], torch.ones(2)*0.02)

    def test_project_depth_units_crop_rays_and_invalid_pixels(self):
        # +X planar depth 2 with direction (1,1,0)/sqrt(2) reaches (2,2,0),
        # then a 90-degree camera yaw and translation maps it to (-1,2,0).
        d = torch.tensor([[2., float("nan"), 2.5, 0.]])
        ray = torch.tensor([1., 1., 0.]) / 2**0.5
        r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        points, valid = perception.depth_points(d, ray.expand(1, 4, 3), torch.zeros(1, 4, 3), r, torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(points[0, 0], torch.tensor([-1., 2., 0.]))
        self.assertEqual(valid.tolist(), [[True, False, False, False]])

    def test_grid_rejects_unknown_gap_and_stair_edge_but_accepts_slope(self):
        grid = perception.SupportGrid(perception.VisualFootholdSettings(), "cpu")
        p = flat_cloud()
        _, _, safe, _, _ = grid.build(p, torch.ones(p.shape[:2], dtype=torch.bool))
        self.assertGreater(grid.sample(safe, torch.tensor([[[0.4, 0.2]]])).item(), 0.5)
        gap = (p[..., 0] > 0.2) & (p[..., 0] < 0.65)
        _, _, unknown_safe, _, _ = grid.build(p, ~gap)
        self.assertEqual(grid.sample(unknown_safe, torch.tensor([[[0.44, 0.2]]])).item(), 0)
        p[..., 2] = torch.where(p[..., 0] < 0.4, 0., -0.16)
        _, _, stairs_safe, _, _ = grid.build(p, torch.ones(p.shape[:2], dtype=torch.bool))
        self.assertEqual(grid.sample(stairs_safe, torch.tensor([[[0.4, 0.2]]])).item(), 0)
        p[..., 2] = 0.2 * p[..., 0]
        _, _, slope_safe, _, _ = grid.build(p, torch.ones(p.shape[:2], dtype=torch.bool))
        self.assertGreater(grid.sample(slope_safe, torch.tensor([[[0.4, 0.2]]])).item(), 0.5)

    def test_curbs_choose_reachable_top_for_short_gap_ground_for_long_gap(self):
        grid = perception.SupportGrid(perception.VisualFootholdSettings(), "cpu")
        p = flat_cloud(2)
        # Platform ends at .20; next begins at .38 (short) / 1.0 (long).
        for e, start in enumerate((0.38, 1.0)):
            top = (p[e, :, 0] < 0.20) | (p[e, :, 0] >= start)
            p[e, :, 2] = torch.where(top, 0., -0.15)
        target, valid, _ = grid.choose(p, torch.ones(p.shape[:2], dtype=torch.bool), torch.zeros(2, 3),
                                      torch.zeros(2, dtype=torch.long), torch.tensor([[1., 0.], [1., 0.]]), torch.ones(2)*0.5)
        self.assertTrue(valid.all())
        self.assertGreater(target[0, 0].item(), 0.38)
        self.assertAlmostEqual(target[0, 2].item(), 0., places=3)
        self.assertAlmostEqual(target[1, 2].item(), -0.15, places=3)

    def test_fk_matches_existing_numpy_demo_with_shuffled_joint_names(self):
        # Existing independent NumPy implementation protects joint ordering,
        # waist ancestry, Jacobian velocity and URDF/shoe frame conventions.
        import numpy as np
        demo = load("foot_state", ROOT / "scripts/foot_state_demo", "numpy_demo")
        urdf = SOURCE.parents[2] / "urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
        names = [j.get("name") for j in ET.parse(urdf).getroot().findall("joint") if j.get("type") != "fixed"][::-1]
        q = torch.rand(3, len(names)) * 0.4 - 0.2
        dq = torch.rand_like(q) * 0.5
        actual = kinematics.LegKinematics(urdf, names, "cpu").compute(q, dq)
        expected = demo.LegKinematics(urdf, names).compute(q.numpy(), dq.numpy())
        for tensor, array in zip(actual, expected):
            np.testing.assert_allclose(tensor.numpy(), array, atol=5e-7)

    def test_contact_confirmation_does_not_invent_initial_touchdown(self):
        observer = kinematics.ContactObserver(2, 0.02, "cpu")
        velocity = torch.zeros(2, 2, 3)
        advance = torch.ones(2, dtype=torch.bool)
        _, touchdown = observer.update(velocity, torch.ones(2, 2)*100, advance)
        self.assertFalse(touchdown.any())
        for _ in range(2):
            observer.update(velocity, torch.zeros(2, 2), advance)
        candidate, touchdown = observer.update(velocity, torch.ones(2, 2)*100, advance)
        self.assertTrue(candidate.all())
        self.assertFalse(touchdown.any())
        _, touchdown = observer.update(velocity, torch.ones(2, 2)*100, advance)
        self.assertTrue(touchdown.all())
        observer.reset(torch.tensor([0]))
        self.assertTrue((observer.state[1] == 2).all())

    def test_commands_lock_reset_isolation_and_support_loss_invalidates_memory(self):
        cfg = perception.VisualFootholdSettings(joint_position_noise=0., joint_velocity_noise=0.)
        cloud = flat_cloud(2)
        core = controller.VisualFootholdController(2, cloud.shape[1], 0.02, cfg, "cpu")
        p = torch.tensor([[[0., 0.1, -0.9], [0., -0.1, -0.9]]]).expand(2, -1, -1).clone()
        r = torch.eye(3).expand(2, 2, 3, 3).clone()
        cloud[..., 2] = -0.9
        v = torch.zeros_like(p)
        gravity = torch.tensor([[0., 0., -1.]]).expand(2, -1)
        force = torch.ones(2, 2)*100
        command = torch.tensor([[0.5, 0., 0.]]).expand(2, -1)
        def step(tick):
            core.update(p, r, v, force, gravity, command, cloud,
                        torch.ones(cloud.shape[:2], dtype=torch.bool), torch.ones(2, dtype=torch.bool), tick)
        for tick in range(6):
            step(tick)
        self.assertTrue(core.valid.all())
        self.assertEqual(core.anchor.tolist(), [1, 1])
        self.assertEqual(tuple(core.observation.shape), (2, 41))
        locked = core.target.clone()
        cloud[..., 2] += 0.08
        step(6)
        torch.testing.assert_close(core.target, locked)
        # Releasing the commanded swing foot must preserve the stance anchor,
        # memory and target, rather than invalidating the command at liftoff.
        force[:, 0] = 0
        step(7)
        self.assertTrue(core.valid.all())
        torch.testing.assert_close(core.target, locked)
        step(8)  # second low-force sample confirms liftoff
        # First candidate error is retained through contact confirmation; a
        # contact reward is emitted once, not once for every standing frame.
        target_body = core.target + p[:, 1]
        p[:, 0] = target_body + torch.tensor([0.02, 0., 0.])
        force[:, 0] = 100
        step(9)
        self.assertFalse(core.last_touchdown.any())
        p[:, 0, 0] += 0.005
        step(10)
        self.assertTrue(core.last_touchdown.all())
        torch.testing.assert_close(core.last_error, torch.ones(2)*0.02, atol=1e-6, rtol=1e-5)
        expected = torch.exp(torch.tensor(-0.02**2 / cfg.target_sigma**2))
        torch.testing.assert_close(core.reward["touchdown"]*0.02, expected.expand(2))
        step(11)
        self.assertFalse(core.last_touchdown.any())
        self.assertTrue((core.reward["touchdown"] == 0).all())
        other_points = core.point_valid[1].clone()
        core.reset(torch.tensor([0]))
        self.assertFalse(core.valid[0])
        torch.testing.assert_close(core.point_valid[1], other_points)
        force[1] = 0
        step(12)
        self.assertFalse(core.valid[1])
        self.assertFalse(core.point_valid[1].any())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
