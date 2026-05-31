"""Modal orchestration for the Cleanup IPPO baseline.

Mirrors `territory/modal_territory_baseline.py` but for the Cleanup env, with
two key differences:

  1. Different IPPO script: `algorithms/IPPO/ippo_cnn_cleanup.py` (env_name
     `clean_up`, NUM_AGENTS=7, NUM_STEPS=1000 by default).
  2. No env-patching needed. Cleanup's `Clean_up.__init__` already accepts
     `inequity_aversion=False` and `svo=False` correctly (unlike Territory's
     `Territory_open.__init__`, which was missing them — `fbbfe2d` bug).

Goal: reproduce the paper's Cleanup finding that common-reward IPPO ≫
individual-reward IPPO (Fig 3 of the SocialJax paper shows common ~1000,
individual ~10 at 1e9 agent-steps).

--- Calibration: 1 run, prints measured cost ---
    modal run cleanup/modal_cleanup_baseline.py

--- Both reward modes × N seeds ---
    modal run cleanup/modal_cleanup_baseline.py --mode sweep --seeds 1

NOTE: run `modal run` from the repo root.
"""

import time

import modal

_GPU_USD_PER_HR = 3.20  # A100-80GB approximate rate.

CUDA_JAX_FIND_LINKS = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

# Same image as the Territory orchestrator — single source of truth for deps.
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

app = modal.App("zkattr-cleanup-baseline")
# Use a separate volume so Cleanup runs don't get mixed up with Territory's.
volume = modal.Volume.from_name("zkattr-cleanup-baseline", create_if_missing=True)

_IPPO_SCRIPT = "/repo/external/SocialJax/algorithms/IPPO/ippo_cnn_cleanup.py"

# reward mode -> ENV_KWARGS.shared_rewards.
_SHARED = {"common": "True", "individual": "False"}


def _patch_socialjax_ippo_cleanup_attribution() -> None:
    """Inject AttributionWrapper around the env in the IPPO Cleanup training path.

    When the IPPO Cleanup subprocess runs with `+ATTRIBUTION=true` (and
    optionally `+ATTRIBUTION_WINDOW=N`), the patched `make_train` wraps the raw
    SocialJax env with `zkattribution.wrappers.AttributionWrapper` using the
    `cleanup_events_batch` v2 beam predicate. The wrapper augments each agent's
    obs with a per-peer cooperation-rate vector α of length `num_agents`,
    updated at every window boundary.

    Important: the patch ONLY wraps the env in `make_train` (the main training
    loop). The `get_rollout` and `evaluate` codepaths are left bare — their
    purpose is debug visualization / GIF generation, not training. Adding the
    wrapper there isn't necessary for the experiment.

    The wrapped env's `observation_space()` reports the augmented shape, so
    the IPPO script's `init_x = jnp.zeros((1, *env.observation_space()[0].shape))`
    automatically picks up the extra channels and the CNN's first conv layer
    infers the right `in_channels` at init time. No network-architecture
    changes needed.

    Patch is idempotent (sentinel-comment check). Applied once per container.
    """
    path = "/repo/external/SocialJax/algorithms/IPPO/ippo_cnn_cleanup.py"
    sentinel = "# zkattr patch: cleanup attribution wrapper"
    with open(path) as f:
        src = f.read()
    if sentinel in src:
        return
    anchor = (
        'def make_train(config):\n'
        '    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])\n'
    )
    injection = (
        'def make_train(config):\n'
        '    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])\n'
        '    # zkattr patch: cleanup attribution wrapper\n'
        '    if config.get("ATTRIBUTION", False):\n'
        '        from zkattribution.wrappers import AttributionWrapper\n'
        '        from zkattribution.predicate import cleanup_events_batch\n'
        '        env = AttributionWrapper(\n'
        '            env, predicate=cleanup_events_batch,\n'
        '            window_size=int(config.get("ATTRIBUTION_WINDOW", 50)),\n'
        '        )\n'
    )
    if anchor not in src:
        raise RuntimeError(
            f"_patch_socialjax_ippo_cleanup_attribution: anchor not found in {path}"
        )
    with open(path, "w") as f:
        f.write(src.replace(anchor, injection, 1))


@app.function(
    # A100-80GB: less contended than 40GB → much lower preemption rate.
    # Documented in territory/docs/premise_findings.md.
    gpu="A100-80GB",
    image=image,
    volumes={"/results": volume},
    timeout=24 * 60 * 60,
)
def run_cell(
    reward_mode: str,
    seed: int,
    total_timesteps: int,
    parameter_sharing: bool = True,
    num_envs: int = 256,
    timeout_minutes: int = 0,
    attribution: bool = False,
    attribution_window: int = 50,
) -> dict:
    """Train one IPPO Cleanup (reward_mode, seed) on a GPU; wandb-offline -> Volume."""
    import os
    import shutil
    import subprocess

    ps_tag = "ps1" if parameter_sharing else "ps0"
    envs_tag = "" if num_envs == 256 else f"_e{num_envs}"
    # Tag the run dir with attribution config so attribution arms land in
    # separate dirs from bare baselines.
    attr_tag = f"_attrV2_w{attribution_window}" if attribution else ""
    run_dir = (
        f"/results/ippo_cleanup_{reward_mode}_t{total_timesteps}"
        f"_{ps_tag}{envs_tag}{attr_tag}_seed{seed}"
    )
    done_marker = os.path.join(run_dir, "RUN_COMPLETE")

    # Idempotent skip — only if a *completed* run exists.
    volume.reload()
    if os.path.exists(done_marker):
        return {"reward_mode": reward_mode, "seed": seed, "minutes": 0.0, "skipped": True}

    # Preserve partial data on retry (same approach as Territory orchestrator).
    if os.path.exists(run_dir):
        backup_dir = f"{run_dir}.backup_{int(time.time())}"
        shutil.move(run_dir, backup_dir)
    os.makedirs(run_dir, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo/external/SocialJax:/repo/src"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = run_dir

    # Patch the IPPO Cleanup script to wrap the env with AttributionWrapper
    # when --attribution is set. No-op if already patched.
    _patch_socialjax_ippo_cleanup_attribution()

    started = time.time()
    ippo_args = [
        "python",
        _IPPO_SCRIPT,
        f"SEED={seed}",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "WANDB_MODE=offline",
        "TUNE=False",
        f"PARAMETER_SHARING={parameter_sharing}",
        f"NUM_ENVS={num_envs}",
        f"ENV_KWARGS.shared_rewards={_SHARED[reward_mode]}",
    ]
    if attribution:
        # `+KEY=VALUE` Hydra prefix adds a key not present in the original yaml.
        # The patched make_train reads these via config.get(...).
        ippo_args.append("+ATTRIBUTION=true")
        ippo_args.append(f"+ATTRIBUTION_WINDOW={attribution_window}")
    proc = subprocess.Popen(
        ippo_args,
        cwd=run_dir,
        env=env,
    )
    if timeout_minutes > 0:
        try:
            proc.wait(timeout=timeout_minutes * 60)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            return {
                "reward_mode": reward_mode,
                "seed": seed,
                "minutes": round((time.time() - started) / 60.0, 1),
                "skipped": False,
                "timed_out": True,
            }
    else:
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, "ippo_cnn_cleanup.py")
    minutes = (time.time() - started) / 60.0

    with open(done_marker, "w") as marker:
        marker.write(f"ippo_cleanup {reward_mode} seed{seed} {minutes:.1f}min\n")
    volume.commit()
    return {
        "reward_mode": reward_mode,
        "seed": seed,
        "minutes": round(minutes, 1),
        "skipped": False,
    }


def _print_run(result) -> None:
    if isinstance(result, BaseException):
        print(f"  ippo_cleanup FAILED -> {type(result).__name__}: {result}")
        return
    if result.get("timed_out"):
        status = f"TIMED OUT after {result['minutes']:.1f} min (watchdog killed)"
    elif result.get("skipped"):
        status = "skipped (already done)"
    else:
        status = f"{result['minutes']:.1f} min"
    print(f"  ippo_cleanup {result['reward_mode']} seed{result['seed']} -> {status}")


@app.local_entrypoint()
def main(
    mode: str = "calibration",
    seeds: int = 1,
    seed: int = 0,
    total_timesteps: int = 300_000_000,
    parameter_sharing: bool = True,
    num_envs: int = 256,
    timeout_minutes: int = 0,
    reward_mode: str = "common",
    attribution: bool = False,
    attribution_window: int = 50,
):
    ps_label = "PS=True" if parameter_sharing else "PS=False (paper config)"
    if mode == "calibration":
        if reward_mode not in _SHARED:
            raise ValueError(f"reward_mode must be one of {list(_SHARED)!r}")
        attr_label = (
            f", attribution V2 (window {attribution_window})" if attribution else ""
        )
        print(
            f"Calibration: 1 IPPO Cleanup run ({reward_mode} reward, seed {seed}) at "
            f"{total_timesteps:,} timesteps, {ps_label}, NUM_ENVS={num_envs}{attr_label} "
            f"on an A100-80GB"
            + (f" (watchdog {timeout_minutes} min)" if timeout_minutes else "") + "...\n"
        )
        result = run_cell.remote(
            reward_mode, seed, total_timesteps, parameter_sharing, num_envs, timeout_minutes,
            attribution, attribution_window,
        )
        _print_run(result)
        if not result.get("skipped") and not result.get("timed_out"):
            per_run = (result["minutes"] / 60.0) * _GPU_USD_PER_HR
            n = 2 * seeds
            print(f"\n  per-run: {result['minutes']:.1f} min  (~${per_run:.2f} on A100-80GB)")
            print(
                f"  full baseline ({n} runs = 2 reward modes x {seeds} seeds): "
                f"~${per_run * n:.2f}"
            )
        return

    if mode == "sweep":
        jobs = [
            (rm, s, total_timesteps, parameter_sharing, num_envs,
             timeout_minutes, attribution, attribution_window)
            for rm in ("common", "individual")
            for s in range(seeds)
        ]
        print(
            f"Cleanup IPPO baseline: {len(jobs)} runs (2 reward modes x {seeds} "
            f"seeds) at {total_timesteps:,} timesteps, {ps_label} on A100-80GB GPUs...\n"
        )
        for result in run_cell.starmap(jobs, return_exceptions=True):
            _print_run(result)
        print("\n  Retrieve:  modal volume get zkattr-cleanup-baseline /<run_dir>")
        return

    raise ValueError(f"unknown mode {mode!r} (expected 'calibration' or 'sweep')")
