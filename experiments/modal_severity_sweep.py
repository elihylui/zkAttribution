"""Modal orchestration for the Track-A severity sweep.

Finds a Cleanup parameter regime where the no-attribution MAPPO baseline
genuinely FAILS to cooperate. That failing baseline is the precondition for
Stage 3 ("does attribution help?") to be a meaningful question at all — the
gate-check baseline cleaned the river even at maximum self-interest, so there
was no cooperation deficit for attribution to close. See
docs/gate_check_findings.md.

Only the no-attribution regime runs here; the sweep is over environment-
dynamics parameters (Phase 1: dirtSpawnProbability). Individual rewards
throughout (shared_rewards=False) — the dilemma-relevant mode.

CALIBRATION-FIRST, like modal_gate_check.py. `modal run` with no args runs the
two extreme cells at a low timestep budget and reports the measured per-run
cost. This confirms the budget AND confirms the scale is long enough to tell a
working baseline apart from a failing one — before committing to the sweep.

--- One-time local setup (if not already done) ---
    uv tool install modal
    modal token new

--- Calibration: 2 runs (dirt 0.5 vs 0.9), prints measured cost ---
    modal run experiments/modal_severity_sweep.py

--- Full Phase-1 sweep: 3 cells x N seeds, only after calibration ---
    modal run experiments/modal_severity_sweep.py --mode sweep --seeds 2

--- A longer scale, if calibration shows 1e7 is too short to differentiate ---
    modal run experiments/modal_severity_sweep.py --mode sweep --seeds 2 --total-timesteps 30000000

--- Retrieve + inspect results locally ---
    modal volume get zkattr-severity-sweep / ./sweep_results
    uv run python scripts/parse_wandb_run.py ./sweep_results/<cell>_t<timesteps>_seed<n>/wandb/offline-run-*
    # Compare `cleaned_water` across cells — look for one that stays low.

NOTE: run `modal run` from the repo root — the image build reads
external/SocialJax/requirements.txt and mounts src/, experiments/,
external/SocialJax/ by relative path.
"""

import time

import modal

# Approximate Modal A100-40GB GPU rate ($/hr) — only for the printed cost
# estimate; the actual bill is whatever Modal charges per-second.
_GPU_USD_PER_HR = 2.10

CUDA_JAX_FIND_LINKS = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# Image: identical to modal_gate_check.py — CUDA 11.8 + cuDNN 8 base, Python
# 3.10, SocialJax's requirements, jaxlib swapped to the CUDA build, repo code
# mounted for runtime.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git")
    .pip_install_from_requirements("external/SocialJax/requirements.txt")
    .run_commands(
        "pip install --force-reinstall --no-deps "
        f"'jaxlib==0.4.23+cuda11.cudnn86' -f {CUDA_JAX_FIND_LINKS}"
    )
    .add_local_dir("src", "/repo/src", ignore=["**/__pycache__"])
    .add_local_dir("experiments", "/repo/experiments", ignore=["**/__pycache__"])
    .add_local_dir(
        "external/SocialJax",
        "/repo/external/SocialJax",
        ignore=["**/.git", "**/__pycache__"],
    )
)

app = modal.App("zkattr-severity-sweep")
volume = modal.Volume.from_name("zkattr-severity-sweep", create_if_missing=True)

# The vanilla (unmodified) SocialJax MAPPO Cleanup script = the no-attribution
# regime.
_NO_ATTR_SCRIPT = "/repo/external/SocialJax/algorithms/MAPPO/mappo_cnn_cleanup.py"

# A "cell" is a set of Cleanup env-param overrides. dirtSpawnProbability is the
# primary severity dial — higher = more pollution per step = more cleaning
# labour needed to keep the river viable. 0.5 is the env default (the
# gate-check regime), which anchors the sweep against a known result.
#
# These keys are NOT present in the MAPPO config yaml, so hydra needs the `+`
# add-key prefix (applied in run_cell). If dirtSpawnProbability alone does not
# break the baseline, add Phase-2 cells here, e.g.
#   "dirt0.9_dep0.2": {"dirtSpawnProbability": 0.9, "thresholdDepletion": 0.2}
SWEEP_CELLS = {
    "dirt0.5": {},
    "dirt0.7": {"dirtSpawnProbability": 0.7},
    "dirt0.9": {"dirtSpawnProbability": 0.9},
}

# Calibration probes the two extremes: if the cheap budget can already tell
# dirt0.5 (should clean) apart from dirt0.9 (should struggle), the scale is
# adequate for the full sweep.
CALIBRATION_CELLS = ["dirt0.5", "dirt0.9"]


@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/results": volume},
    timeout=6 * 60 * 60,
)
def run_cell(cell_id: str, cell_kwargs: dict, seed: int, total_timesteps: int) -> dict:
    """Train one no-attribution (cell, seed) on a GPU; wandb-offline -> Volume."""
    import os
    import shutil
    import subprocess

    # The timestep budget is part of the dir name so the same cell run at
    # different scales (e.g. a 1e7 calibration vs a 3e7 sweep) never collides
    # or falsely skips via the RUN_COMPLETE sentinel.
    run_dir = f"/results/{cell_id}_t{total_timesteps}_seed{seed}"
    done_marker = os.path.join(run_dir, "RUN_COMPLETE")

    # Idempotent skip — ONLY if a *completed* run exists. The sentinel is
    # written after training fully succeeds; a bare wandb/ dir is a preempted
    # partial and must NOT count as done.
    volume.reload()
    if os.path.exists(done_marker):
        return {"cell": cell_id, "seed": seed, "minutes": 0.0, "skipped": True}

    # Fresh run: wipe any partial state left by a preempted attempt.
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo/external/SocialJax:/repo/src"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = run_dir

    # Env-dynamics overrides. These keys are absent from the MAPPO config yaml,
    # so hydra requires the `+` add-key prefix. (shared_rewards IS in the yaml,
    # so it stays a plain override below.)
    cell_args = [f"+ENV_KWARGS.{k}={v}" for k, v in cell_kwargs.items()]

    started = time.time()
    # cwd = run_dir (on the writable Volume) so checkpoints/evaluation/wandb
    # land there; the script path is absolute so cwd does not affect imports.
    subprocess.run(
        [
            "python",
            _NO_ATTR_SCRIPT,
            f"SEED={seed}",
            f"TOTAL_TIMESTEPS={total_timesteps}",
            "NUM_ENVS=64",
            "WANDB_MODE=offline",
            "ENV_KWARGS.shared_rewards=False",
            *cell_args,
        ],
        cwd=run_dir,
        env=env,
        check=True,
    )
    minutes = (time.time() - started) / 60.0

    # Sentinel: written only after the subprocess fully succeeded, so a
    # preemption + restart re-trains instead of falsely skipping.
    with open(done_marker, "w") as marker:
        marker.write(f"{cell_id} seed{seed} {minutes:.1f}min\n")
    volume.commit()
    return {"cell": cell_id, "seed": seed, "minutes": round(minutes, 1), "skipped": False}


def _print_run(result: dict) -> None:
    status = (
        "skipped (already done)"
        if result.get("skipped")
        else f"{result['minutes']:.1f} min"
    )
    print(f"  {result['cell']} seed{result['seed']} -> {status}")


@app.local_entrypoint()
def main(mode: str = "calibration", seeds: int = 2, total_timesteps: int = 10_000_000):
    if mode == "calibration":
        jobs = [(c, SWEEP_CELLS[c], 0, total_timesteps) for c in CALIBRATION_CELLS]
        print(
            f"Calibration: {len(jobs)} runs ({', '.join(CALIBRATION_CELLS)}) at "
            f"{total_timesteps:,} timesteps on A100 GPUs...\n"
        )
        results = list(run_cell.starmap(jobs))
        for result in results:
            _print_run(result)

        measured = [r["minutes"] for r in results if not r.get("skipped")]
        if measured:
            avg = sum(measured) / len(measured)
            per_run_usd = (avg / 60.0) * _GPU_USD_PER_HR
            n_sweep = len(SWEEP_CELLS) * seeds
            print(f"\n  per-run: ~{avg:.1f} min  (~${per_run_usd:.2f} on an A100)")
            print(
                f"  full sweep estimate ({len(SWEEP_CELLS)} cells x {seeds} seeds "
                f"= {n_sweep} runs): ~${per_run_usd * n_sweep:.2f}"
            )
        print("\n  Retrieve + compare cleaned_water across cells:")
        print("    modal volume get zkattr-severity-sweep / ./sweep_results")
        print(
            "  If this scale cannot separate dirt0.5 from dirt0.9, re-run the "
            "sweep with a larger --total-timesteps."
        )
        return

    if mode == "sweep":
        jobs = [
            (c, SWEEP_CELLS[c], s, total_timesteps)
            for c in SWEEP_CELLS
            for s in range(seeds)
        ]
        print(
            f"Severity sweep: {len(jobs)} runs ({len(SWEEP_CELLS)} cells x "
            f"{seeds} seeds) at {total_timesteps:,} timesteps on A100 GPUs...\n"
        )
        for result in run_cell.starmap(jobs):
            _print_run(result)
        print(
            "\n  Retrieve results:  "
            "modal volume get zkattr-severity-sweep / ./sweep_results"
        )
        return

    raise ValueError(f"unknown mode {mode!r} (expected 'calibration' or 'sweep')")
