"""Modal orchestration for the Stage 3/4 cloud gate-check.

Runs the three non-crypto Cleanup regimes (no-attribution / oracle / self-
reported) at 1e8 timesteps on rented GPUs, to answer:
  - Stage 3 gate: does oracle attribution improve cooperation over no-attribution?
  - Stage 4: do agents inflate self-reported claims?

CALIBRATION-FIRST. The default `modal run` does ONE run and reports the real
per-run time + cost. The full multi-seed sweep is a separate explicit command,
so a spending cap can't be blown by accident.

--- One-time local setup ---
    uv tool install modal
    modal token new
    # In the Modal dashboard, set a workspace spending limit (e.g. $10).

--- Calibration: one run, prints measured cost (~$1-3) ---
    modal run experiments/modal_gate_check.py

--- Full sweep: only after calibration confirms the per-run cost ---
    modal run experiments/modal_gate_check.py --mode sweep --seeds 2

--- Retrieve results locally ---
    modal volume get zkattr-gate-check / ./modal_results
    uv run python scripts/parse_wandb_run.py ./modal_results/<regime>_seed<n>/wandb/offline-run-*

NOTE: run `modal run` from the repo root — the image build reads
`external/SocialJax/requirements.txt` and mounts `src/`, `experiments/`,
`external/SocialJax/` by relative path.
"""

import time

import modal

# Approximate Modal A100-40GB GPU rate ($/hr) — only for the printed cost
# estimate; the actual bill is whatever Modal charges per-second.
_GPU_USD_PER_HR = 2.10

CUDA_JAX_FIND_LINKS = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# Reward mode. The brief/proposal want a real social dilemma — agents
# individually tempted to free-ride — which is INDIVIDUAL rewards. SocialJax's
# MAPPO config defaults to shared (team) rewards, which mutes the dilemma; we
# override it via the hydra arg `ENV_KWARGS.shared_rewards`.
SHARED_REWARDS = False
REWARD_TAG = "shared" if SHARED_REWARDS else "individual"

# Image: CUDA 11.8 + cuDNN 8 base (mirrors the GPU setup SocialJax's README
# documents), Python 3.10, all deps from SocialJax's requirements.txt, then
# jaxlib swapped to the CUDA build. The repo code is mounted for runtime.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git")
    .pip_install_from_requirements("external/SocialJax/requirements.txt")
    .run_commands(
        # Replace the CPU jaxlib with the CUDA build (jax stays 0.4.23).
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

app = modal.App("zkattr-gate-check")
volume = modal.Volume.from_name("zkattr-gate-check", create_if_missing=True)

_SCRIPTS = {
    "no_attribution": "/repo/external/SocialJax/algorithms/MAPPO/mappo_cnn_cleanup.py",
    "oracle": "/repo/experiments/train_mappo_attribution.py",
    "self_reported": "/repo/experiments/train_mappo_self_reported.py",
}


@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/results": volume},
    timeout=6 * 60 * 60,
)
def run_regime(regime: str, seed: int, total_timesteps: int) -> dict:
    """Train one (regime, seed) on a GPU; wandb-offline output -> the Volume."""
    import os
    import shutil
    import subprocess

    # Reward mode is in the dir name so shared/individual runs never collide.
    run_dir = f"/results/{regime}_{REWARD_TAG}_seed{seed}"
    done_marker = os.path.join(run_dir, "RUN_COMPLETE")

    # Idempotent skip — ONLY if a *completed* run exists. The sentinel below is
    # written after training fully succeeds; a bare wandb/ dir is not enough
    # (a preempted attempt leaves a partial one, which would falsely "skip").
    volume.reload()
    if os.path.exists(done_marker):
        return {"regime": regime, "seed": seed, "minutes": 0.0, "skipped": True}

    # Fresh run: wipe any partial state left by a preempted attempt.
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo/external/SocialJax:/repo/src"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = run_dir

    started = time.time()
    # cwd = run_dir (on the writable Volume) so checkpoints/evaluation/wandb
    # land there; the script path is absolute so cwd does not affect imports.
    subprocess.run(
        [
            "python",
            _SCRIPTS[regime],
            f"SEED={seed}",
            f"TOTAL_TIMESTEPS={total_timesteps}",
            "NUM_ENVS=64",
            "WANDB_MODE=offline",
            f"ENV_KWARGS.shared_rewards={SHARED_REWARDS}",
        ],
        cwd=run_dir,
        env=env,
        check=True,
    )
    minutes = (time.time() - started) / 60.0

    # Sentinel: written only after the subprocess fully succeeded, so a
    # preemption + restart re-trains instead of falsely skipping.
    with open(done_marker, "w") as marker:
        marker.write(f"{regime} seed{seed} {minutes:.1f}min\n")
    volume.commit()
    return {"regime": regime, "seed": seed, "minutes": round(minutes, 1), "skipped": False}


@app.local_entrypoint()
def main(mode: str = "calibration", seeds: int = 2, total_timesteps: int = 100_000_000):
    if mode == "calibration":
        print(f"Calibration: one no-attribution run ({REWARD_TAG} rewards) at "
              f"{total_timesteps:,} timesteps on an A100 GPU...\n")
        result = run_regime.remote("no_attribution", 0, total_timesteps)
        minutes = result["minutes"]
        per_run_usd = (minutes / 60.0) * _GPU_USD_PER_HR
        print(f"\n  per-run: {minutes:.1f} min  (~${per_run_usd:.2f} on an A100)")
        print(f"  full sweep estimate (3 regimes x {seeds} seeds = {3 * seeds} runs): "
              f"~${per_run_usd * 3 * seeds:.2f}")
        print("\n  If that fits the budget, launch the sweep:")
        print(f"    modal run experiments/modal_gate_check.py --mode sweep --seeds {seeds}")
        return

    if mode == "sweep":
        regimes = ["no_attribution", "oracle", "self_reported"]
        jobs = [(r, s, total_timesteps) for r in regimes for s in range(seeds)]
        print(f"Sweep: {len(jobs)} runs ({len(regimes)} regimes x {seeds} seeds) "
              "in parallel on A100 GPUs...\n")
        for result in run_regime.starmap(jobs):
            status = "skipped (already done)" if result.get("skipped") else (
                f"{result['minutes']:.1f} min")
            print(f"  {result['regime']} seed{result['seed']} -> {status}")
        print("\n  Retrieve results:  modal volume get zkattr-gate-check / ./modal_results")
        return

    raise ValueError(f"unknown mode {mode!r} (expected 'calibration' or 'sweep')")
