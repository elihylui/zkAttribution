"""Modal orchestration for the self-interest (S) sweep on the hardened env.

The Track-A severity sweep found dirt0.7 — a Cleanup regime where the
no-attribution baseline fails at maximum self-interest (S=1). This sweep dials
S, the Willis self-interest level (`RewardExchangeWrapper`), on that fixed env.

Phase 1 (the s-sweep): no-attribution baseline, dirt0.7, S below 1.0. Maps the
cooperation cliff S* and confirms dirt0.7 is a genuine dilemma — does the
baseline cooperate at low S (heavy reward-sharing)? If even S=1/7 fails,
dirt0.7 is physically impossible rather than a social dilemma.

Phase 2 (Stage 3): no-attribution vs oracle attribution at chosen severity
points (S-values picked from Phase 1's curve), 2 seeds — the Stage-3 gate.

Runs at 3e7 timesteps (proven sufficient by the Track-A sweep). Individual base
rewards (shared_rewards=False); RewardExchangeWrapper then applies S.

--- Phase 1: no-attribution s-sweep (6 runs, ~$9) ---
    modal run experiments/modal_self_interest_sweep.py --mode phase1

--- Phase 2: Stage-3 grid (fill PHASE2_S_VALUES from Phase 1's result first) ---
    modal run experiments/modal_self_interest_sweep.py --mode phase2 --seeds 2

--- Retrieve results (per run dir) ---
    cd <some dir> && modal volume get zkattr-self-interest-sweep /<run_dir>
    uv run python scripts/parse_wandb_run.py <run_dir>

NOTE: run `modal run` from the repo root — the image build reads
external/SocialJax/requirements.txt by relative path.
"""

import time

import modal

# Approximate Modal A100-40GB rate ($/hr) — for the printed estimate only.
_GPU_USD_PER_HR = 2.10

CUDA_JAX_FIND_LINKS = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# Image: identical to modal_severity_sweep.py.
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

app = modal.App("zkattr-self-interest-sweep")
volume = modal.Volume.from_name("zkattr-self-interest-sweep", create_if_missing=True)

# The generalized training script: ATTRIBUTION toggles the regime, S sets the
# Willis self-interest level.
_TRAIN_SCRIPT = "/repo/experiments/train_mappo_attribution.py"

# Hardened env from the Track-A severity sweep: at dirtSpawnProbability 0.7 the
# no-attribution baseline fails at S=1.
DIRT_SPAWN = 0.7

# Phase 1 — no-attribution baseline, S swept below 1.0. (S=1.0 is already known
# to fail, from the Track-A sweep, so it is not re-run here.) 1/7 ~= 0.14 is the
# fully-utilitarian floor of the Willis range.
PHASE1_S_VALUES = [0.85, 0.7, 0.55, 0.4, 0.28, 0.14]

# Phase 2 — the Stage-3 severity points, chosen from Phase 1's cliff. Fill this
# in (e.g. [0.85, 0.5, 0.2]) after Phase 1, then run --mode phase2.
PHASE2_S_VALUES = []  # list of floats

# regime -> the ATTRIBUTION flag passed to the training script.
_ATTRIBUTION = {"no_attribution": "false", "oracle": "true"}


@app.function(
    gpu="A100-40GB",
    image=image,
    volumes={"/results": volume},
    timeout=6 * 60 * 60,
)
def run_cell(regime: str, s: float, seed: int, total_timesteps: int) -> dict:
    """Train one (regime, S, seed) on a GPU; wandb-offline output -> the Volume."""
    import os
    import shutil
    import subprocess

    # regime / S / timestep budget all in the dir name so runs never collide
    # or falsely skip via the RUN_COMPLETE sentinel.
    s_tag = f"{s:.2f}".replace(".", "")
    run_dir = f"/results/{regime}_s{s_tag}_t{total_timesteps}_seed{seed}"
    done_marker = os.path.join(run_dir, "RUN_COMPLETE")

    volume.reload()
    if os.path.exists(done_marker):
        return {"regime": regime, "s": s, "seed": seed, "minutes": 0.0, "skipped": True}

    # Fresh run: wipe any partial state left by a preempted attempt.
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo/external/SocialJax:/repo/src"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = run_dir

    started = time.time()
    subprocess.run(
        [
            "python",
            _TRAIN_SCRIPT,
            f"SEED={seed}",
            f"TOTAL_TIMESTEPS={total_timesteps}",
            "NUM_ENVS=64",
            "WANDB_MODE=offline",
            "ENV_KWARGS.shared_rewards=False",
            f"+ENV_KWARGS.dirtSpawnProbability={DIRT_SPAWN}",
            f"+ATTRIBUTION={_ATTRIBUTION[regime]}",
            f"+S={s}",
        ],
        cwd=run_dir,
        env=env,
        check=True,
    )
    minutes = (time.time() - started) / 60.0

    # Sentinel: written only after the subprocess fully succeeded.
    with open(done_marker, "w") as marker:
        marker.write(f"{regime} s{s} seed{seed} {minutes:.1f}min\n")
    volume.commit()
    return {
        "regime": regime,
        "s": s,
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
    print(f"  {result['regime']} S={result['s']} seed{result['seed']} -> {status}")


@app.local_entrypoint()
def main(mode: str = "phase1", seeds: int = 2, total_timesteps: int = 30_000_000):
    if mode == "phase1":
        jobs = [("no_attribution", s, 0, total_timesteps) for s in PHASE1_S_VALUES]
        print(
            f"Phase 1 — no-attribution s-sweep on dirt{DIRT_SPAWN}: {len(jobs)} runs "
            f"(S = {PHASE1_S_VALUES}) at {total_timesteps:,} timesteps...\n"
        )
        results = list(run_cell.starmap(jobs))
        for result in results:
            _print_run(result)
        measured = [r["minutes"] for r in results if not r.get("skipped")]
        if measured:
            avg = sum(measured) / len(measured)
            print(
                f"\n  per-run ~{avg:.1f} min  "
                f"(~${(avg / 60.0) * _GPU_USD_PER_HR:.2f} on an A100)"
            )
        print(
            "\n  Retrieve + parse each run, then find the S where cleaned_water / "
            "return recovers — that brackets the cooperation cliff S*."
        )
        return

    if mode == "phase2":
        if not PHASE2_S_VALUES:
            raise ValueError(
                "PHASE2_S_VALUES is empty — fill it with the severity points "
                "chosen from Phase 1's result, then re-run."
            )
        regimes = ["no_attribution", "oracle"]
        jobs = [
            (regime, s, seed, total_timesteps)
            for regime in regimes
            for s in PHASE2_S_VALUES
            for seed in range(seeds)
        ]
        print(
            f"Phase 2 — Stage-3 grid on dirt{DIRT_SPAWN}: {len(jobs)} runs "
            f"({len(regimes)} regimes x {len(PHASE2_S_VALUES)} S-values x {seeds} "
            f"seeds) at {total_timesteps:,} timesteps...\n"
        )
        for result in run_cell.starmap(jobs):
            _print_run(result)
        print("\n  Retrieve:  modal volume get zkattr-self-interest-sweep /<run_dir>")
        return

    raise ValueError(f"unknown mode {mode!r} (expected 'phase1' or 'phase2')")
