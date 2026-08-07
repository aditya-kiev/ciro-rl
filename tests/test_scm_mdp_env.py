"""Unit tests for the SCM-MDP environment generator (CIRO Appendix B.1).

Checks that resets/steps (via the generator API) produce correctly-shaped
observations, renders, rewards and that interventions work as do().
"""

from __future__ import annotations

import torch

from ciro_rl.envs.scm_mdp import get_scm_mdp


def test_render_shape():
    for kind in ("independent", "causal_chain", "confounded"):
        scm = get_scm_mdp(kind, latent_dim=8, seed=7)
        z = scm.sample_prior(4)
        assert z.shape == (4, 8)
        img = scm.render(z)
        assert img.shape == (4, 3, 32, 32)
        assert torch.isfinite(img).all()


def test_offline_trajectory_shapes():
    for kind in ("independent", "causal_chain", "confounded"):
        scm = get_scm_mdp(kind, latent_dim=8, seed=7)
        lat, act, rew = scm.build_offline_trajectories(3, 10, seed=0)
        assert lat.shape == (3, 10, 8)
        assert act.shape == (3, 10, 1)
        assert rew.shape == (3, 10)
        assert torch.isfinite(lat).all()
        assert torch.isfinite(rew).all()


def test_offline_trajectories_deterministic_in_seed():
    scm = get_scm_mdp("confounded", latent_dim=8, seed=7)
    l1, a1, r1 = scm.build_offline_trajectories(8, 20, seed=3)
    l2, a2, r2 = scm.build_offline_trajectories(8, 20, seed=3)
    assert torch.equal(l1, l2)
    assert torch.equal(a1, a2)
    assert torch.equal(r1, r2)


def test_intervention_sets_coordinate():
    scm = get_scm_mdp("independent", latent_dim=8, seed=7)
    z = scm.sample_prior(6)
    zi = scm.intervene(z, 2, 5.0)
    assert torch.allclose(zi[:, 2], torch.full((6,), 5.0))
    others = torch.arange(8) != 2
    assert torch.allclose(zi[:, others], z[:, others])
    # original untouched
    assert not torch.allclose(zi[:, 2], z[:, 2])


def test_reward_is_finite():
    scm = get_scm_mdp("causal_chain", latent_dim=8, seed=7)
    z = scm.sample_prior(5)
    r = scm.reward(z)
    assert r.shape == (5,)
    assert torch.isfinite(r).all()


def test_rollout_preserves_shape():
    scm = get_scm_mdp("confounded", latent_dim=8, seed=7,
                      confounded_pair=(0, 1))
    z0 = scm.sample_prior(8)
    act = torch.rand(8, 5, 1) * 2 - 1
    noise = scm.new_noise(8, 5)
    zT = scm.rollout(z0, act, noise)
    assert zT.shape == (8, 8)
    assert torch.isfinite(zT).all()


def test_rollout_shared_noise_is_deterministic_given_noise():
    """Original vs do-intervened rollouts must share the same noise stream."""
    scm = get_scm_mdp("confounded", latent_dim=8, seed=7, confounded_pair=(0, 1))
    z0 = scm.sample_prior(8)
    act = torch.rand(8, 4, 1) * 2 - 1
    noise = scm.new_noise(8, 4)
    zi = scm.intervene(z0, 3, 2.0)  # intervene a causal parent not in (0,1)
    z_ref = scm.rollout(z0, act, noise)
    z_int = scm.rollout(zi, act, noise)
    assert not torch.allclose(z_ref, z_int)
    # latents 4-7 are nuisance (no cross coupling): identical under shared noise
    persist = [4, 5, 6, 7]
    assert torch.allclose(z_ref[:, persist], z_int[:, persist])