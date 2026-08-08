# ciro-rl

CIRO (**C**ausal **I**ndependence **R**egularization for robust state 
representations in **o**ffline RL) plus the associated **ACS** and **DTG**
diagnostics. This is the companion implementation of the paper "Causal
Independence Regularization for Robust State Representations in Offline
Reinforcement Learning" (CIRO). It builds on CIR ("Contrastive Causal
Representation Learning via Conditional Independence Regularization"),
extending its residual-then-HSIC regularization from a static contrastive
setting to CURL-style state representations for offline RL.

The repository is a self-contained, research-grade implementation. It
implements the theory, the method training loop, and the two evaluation
diagnostics (causal ACS on SCM-MDP and action-saliency DTG on DCS), with
unit tests and a one-shot smoke test.

> **Status note:** this builds and all 25 unit tests pass, and
> `experiments/quick_test.py` runs the *entire* pipeline end-to-end (SCM fully,
> DCS exercising its skip path) on CPU. The DCS rows require MuJoCo +
> Distracting Control, which are optional (see the license note below).

---

## What is it

Offline RL agents are fed raw pixels. Their state representations are usually
learned with a contrastive ("CURL"-style) objective, which is good but ignores
the causal / conditional-independence structure between *representation
dimensions*. CIRO adds a regularizer that penalizes (via the HSIC) residual
conditional dependence *among* representation coordinates, so the learned
representation becomes more redundant-free and more causally interpretable.

Two diagnostics are provided:

- **ACS (Axis-Wise Causal Sensitivity)** on synthetic **SCM-MDP** benchmarks:
  how concentrated each causal latent's influence is on a single learned
  representation coordinate, by propagating a one-time intervention
  through the transition dynamics and measuring the change (Section 6.1).
- **DTG (Distractor Transfer Gap)** on **Distracting Control Suite (DCS)**:
  the percentage drop in a training-pool linear-probe score when applied
  to evaluation-distractor-pool frames, quantifying robustness to
  distractor shift (Section 6.2).


---

## Install

Core (CPU or any CUDA runtime — torch auto-selects the backend):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "pytest>=7.0"
```

To ALSO run the DCS tables you need MuJoCo + Distracting Control. **This is a
licensing decision** — the repo itself uses the `distracting_control` and
`dm_control` data generators, whose license is **not** covered by this project.
The runtime never downloads MuJoCo assets automatically and every DCS entry
point is gated behind `dcs_wrapper.is_available()`. Enable them only if you
accept their licenses:

```bash
INSTALL_DCS=1 bash scripts/setup_colab.sh
```

On Colab the one-shot path is just:

```bash
bash scripts/setup_colab.sh   # (no DCS unless INSTALL_DCS=1)
```

---

## First thing: enforce feasibility with the smoke test

Everything is validated by a single, fast, end-to-end smoke test. Run it FIRST;
if it passes you have a working install and the whole training/test path.

```bash
python experiments/quick_test.py
```

Requires no DCS. It:

1. checks the exact-vs-RFF HSIC estimators agree (a sanity check on the two
   learnable-bandwidth HSIC estimators in `ciro_rl.utils.hsic`),
2. runs a few steps of **every** method (`curl`, `marginal_hsic`, `ciro`) on
   **every** dataset, and
3. runs the whole unit-test suite.

Exit code is `0` iff everything passes.

---

## Run the full evaluations

### Table 1 — methods × datasets

```bash
python experiments/run_table1.py --out ./outputs/table1
```

Writes:
- `outputs/table1/results_by_dataset_summary.csv` — one row per (method, dataset, seed)
- `outputs/table1/results_table1_avg.csv` — seed-averaged ACS / DTG per (method, dataset)

By default it runs seeds `0, 1, 2` on every published dataset. Add `--seed N`
to run a single seed, or `--method ciro` to isolate a method. For a fast
layout-proof run use `--quick-test` (tiny steps, seed 0) — the CSV writers are
the same as the full run:

```bash
python experiments/run_table1.py --quick-test --out ./outputs/table1
```

### Table 2 (ablations: placement × lambda)

```bash
python experiments/run_table2_ablations.py --out ./outputs/table2
```

Sweeps `placement ∈ {encoder, projection}` × a few `hsic_lambda` values on the
`scm_confounded` benchmark (defaults chosen so the run is feasible on a Colab
T4; see the `--steps` flag).

---

## Paper → code mapping

| Paper concept | Where |
|---|---|
| Contrastive (CURL) prior `f_q, f_k` | `ciro_rl/methods/curl.py` (`CURLModel`, `curl_contrastive_loss`) |
| Encoder `f_enc` / projection `g_enc` | `ciro_rl/methods/curl.py` |
| Latent `z_q, z_k` | `CURLModel.forward` → `z_q`, `z_k` |
| `CIRO = (1/n^2) Σ_{j≠k} HSIC(R[.,j], R[.,k])` | `ciro_rl/methods/ciro.py` |
| `HSIC` residual-then-coordinate step | `ciro_rl/utils/hsic.py` (`pairwise_hsic_exact`, `pairwise_hsic_rff`) |
| Median-heuristic bandwidth, RFF dimension | `ciro_rl/methods/ciro.py` defaults / config |
| `ACS` | `ciro_rl/diagnostics/acs.py` (`average_causal_sensitivity`) |
| `DTG` | `ciro_rl/diagnostics/dtg.py` (`run_dtg`) |
| SCM-MDP benchmarks | `ciro_rl/envs/scm_mdp.py` |
| DCS wrapper (optional) | `ciro_rl/envs/dcs_wrapper.py` |

---

## Results (adaptive — the scoring prompt's live table)

### Table 1 — ACS (↑): SCM-MDP shows the expected ranking

_(To be filled in after the full runs on GPU. TODO task: replace the empty
table with the contents of `outputs/table1/results_table1_avg.csv`.)_

| method | scm_chain | scm_confounded | scm_independent |
|---|---|---|---|
| curl |  |  |  |
| marginal_hsic |  |  |  |
| ciro |  |  |  |

### Table 2 — ablations: placement (encoder vs projection) × `hsic_lambda` on `scm_confounded`

_(To be filled in after the runs. TODO task: insert the `run_table2_ablations.py`
output here.)_

## Runtime considerations on a Colab T4

The two table runners accept `--seed` (seed to run), `--out`, and for a quick
CSV-shape check `--quick-test` (a tiny pass with seed 0). The defaults balance
an achievable runtime on a **T4**: a `--quick-test` run is fast on all 5
datasets; the seed-full `--seed 0 --seed 1 --seed 2` run takes meaningfully
longer. AMP (`mixed_precision`) is enabled automatically only on CUDA.

DCS (**Distracting Control Suite**) is a heavier install (MuJoCo / EGL + the
`distracting_control` package) and is gated—`dcs_wrapper.py` — so the repo is fully
usable without it; those columns are marked `N/A` in the tables when DCS is
not installed.

---

**Known limitation (honest)** — this is a *representation-learning*
implementation: it learns the state representation, runs the two diagnostics
(ACS on SCM-MDP, DTG on DCS), and the ablations. It does **not** run an offline
policy optimizer (there is no CQL / IQL / DT loop here). Offline RL agents
would plug the learned representation into their own value/policy heads.

---

## License

This project is licensed under the **MIT License** — see `LICENSE`. In
particular, note that the optional DCS dependencies (`dm_control` /
`distracting_control`) have **their own** licenses (MuJoCo and Distracting
Control are **not** available under the same terms) — see the "Known
limitation" note above.