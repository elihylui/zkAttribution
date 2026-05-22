"""Modal orchestration for the Phase-1a Territory IPPO baseline.

Reproduces SocialJax's IPPO Territory result — common-reward and
individual-reward IPPO on `territory_open` — to (1) confirm the paper's finding
that common-reward IPPO fails Territory due to credit assignment, and (2) get
the baseline returns that the Phase-1 gate compares against the cooperative
ceiling.

The no-attribution baseline is `ippo_cnn_territory_open.py` off the shelf — the
paper's exact setup. Horizon 3e8 (the config default). Territory is the slowest
SocialJax env, so per-run cost is measured before the full sweep.

CALIBRATION-FIRST. `modal run` with no args does ONE run (common reward, 1
seed, full 3e8) and reports the measured per-run time + cost.

--- Calibration: 1 run, prints measured cost ---
    modal run territory/modal_territory_baseline.py

--- Full baseline: both reward modes x N seeds, after calibration ---
    modal run territory/modal_territory_baseline.py --mode sweep --seeds 3

--- Retrieve + inspect ---
    modal volume get zkattr-territory-baseline /<run_dir>
    uv run python scripts/parse_wandb_run.py <run_dir>

NOTE: run `modal run` from the repo root — the image build reads
external/SocialJax/requirements.txt by relative path.
"""

import time

import modal

# Approximate Modal A100-40GB rate ($/hr) — for the printed estimate only.
_GPU_USD_PER_HR = 2.10

CUDA_JAX_FIND_LINKS = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# Image: same CUDA 11.8 + cuDNN 8 base as the other orchestrators.
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
    .add_local_dir(
        "external/SocialJax",
        "/repo/external/SocialJax",
        ignore=["**/.git", "**/__pycache__"],
    )
)

app = modal.App("zkattr-territory-baseline")
volume = modal.Volume.from_name("zkattr-territory-baseline", create_if_missing=True)

# The off-the-shelf SocialJax IPPO Territory script = the no-attribution baseline.
_IPPO_SCRIPT = "/repo/external/SocialJax/algorithms/IPPO/ippo_cnn_territory_open.py"

# reward mode -> ENV_KWARGS.shared_rewards. "common" (team reward) is the regime
# the SocialJax paper shows fails Territory (credit assignment); "individual" is
# the paper's stronger baseline.
_SHARED = {"common": "True", "individual": "False"}


@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/results": volume},
    timeout=12 * 60 * 60,
)
def run_cell(reward_mode: str, seed: int, total_timesteps: int) -> dict:
    """Train one IPPO Territory (reward_mode, seed) on a GPU; wandb-offline -> Volume."""
    import os
    import shutil
    import subprocess

    run_dir = f"/results/ippo_territory_{reward_mode}_t{total_timesteps}_seed{seed}"
    done_marker = os.path.join(run_dir, "RUN_COMPLETE")

    # Idempotent skip — only if a *completed* run exists (sentinel written after
    # the subprocess fully succeeds; a bare wandb/ dir is a preempted partial).
    volume.reload()
    if os.path.exists(done_marker):
        return {"reward_mode": reward_mode, "seed": seed, "minutes": 0.0, "skipped": True}

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo/external/SocialJax:/repo/src"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = run_dir

    started = time.time()
    # TUNE=False -> single_run (the config ships TUNE=True for a hyperparam
    # sweep). cwd=run_dir so checkpoints/eval/wandb land on the Volume.
    subprocess.run(
        [
            "python",
            _IPPO_SCRIPT,
            f"SEED={seed}",
            f"TOTAL_TIMESTEPS={total_timesteps}",
            "WANDB_MODE=offline",
            "TUNE=False",
            f"ENV_KWARGS.shared_rewards={_SHARED[reward_mode]}",
        ],
        cwd=run_dir,
        env=env,
        check=True,
    )
    minutes = (time.time() - started) / 60.0

    with open(done_marker, "w") as marker:
        marker.write(f"ippo_territory {reward_mode} seed{seed} {minutes:.1f}min\n")
    volume.commit()
    return {
        "reward_mode": reward_mode,
        "seed": seed,
        "minutes": round(minutes, 1),
        "skipped": False,
    }


def _print_run(result: dict) -> None:
    status = (
        "skipped (already done)"
        if result.get("skipped")
        else f"{result['minutes']:.1f} min"
    )
    print(f"  ippo_territory {result['reward_mode']} seed{result['seed']} -> {status}")


@app.local_entrypoint()
def main(mode: str = "calibration", seeds: int = 3, total_timesteps: int = 300_000_000):
    if mode == "calibration":
        print(
            f"Calibration: 1 IPPO Territory run (common reward, seed 0) at "
            f"{total_timesteps:,} timesteps on an A100...\n"
        )
        result = run_cell.remote("common", 0, total_timesteps)
        _print_run(result)
        if not result.get("skipped"):
            per_run = (result["minutes"] / 60.0) * _GPU_USD_PER_HR
            n = 2 * seeds
            print(f"\n  per-run: {result['minutes']:.1f} min  (~${per_run:.2f} on an A100)")
            print(
                f"  full baseline ({n} runs = 2 reward modes x {seeds} seeds): "
                f"~${per_run * n:.2f}"
            )
        print("\n  If that fits, launch the full baseline:")
        print(
            f"    modal run territory/modal_territory_baseline.py --mode sweep "
            f"--seeds {seeds}"
        )
        return

    if mode == "sweep":
        jobs = [
            (rm, s, total_timesteps)
            for rm in ("common", "individual")
            for s in range(seeds)
        ]
        print(
            f"Territory IPPO baseline: {len(jobs)} runs (2 reward modes x {seeds} "
            f"seeds) at {total_timesteps:,} timesteps on A100 GPUs...\n"
        )
        for result in run_cell.starmap(jobs):
            _print_run(result)
        print("\n  Retrieve:  modal volume get zkattr-territory-baseline /<run_dir>")
        return

    raise ValueError(f"unknown mode {mode!r} (expected 'calibration' or 'sweep')")
