"""Fast A100 launcher for the Cleanup self-report / sabotage EVALS.

Runs any eval mode of `cleanup/train_selfreport_exploit.py` (or `scripts/*.py`) on
an A100, reading checkpoints straight from the `zkattr-cleanup-baseline` volume —
so the rollout sweeps that take 20–40 min on local CPU finish in a few minutes.

`{sc}` and `{recip}` in the CLI string resolve to the on-volume checkpoint dirs:
  sc    = 858 aggregate (collapse_spread) individual+α policy
  recip = common+α (per-agent) learned reciprocator

Examples (note: quote the --cli string):
    modal run cleanup/modal_eval.py --cli \
      "cleanup/train_selfreport_exploit.py --sabotage --ckpt-dir {recip} --focal-slot 1 --episodes 32"
    modal run cleanup/modal_eval.py --cli \
      "cleanup/train_selfreport_exploit.py --sweep-claim --ckpt-dir {recip} --per-agent --episodes 16"
    modal run cleanup/modal_eval.py --cli \
      "scripts/ablate_alpha_cleanup.py --attr-dir {recip} --episodes 24"
"""
import modal

CUDA_JAX_FIND_LINKS = "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

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
    .add_local_dir("external/SocialJax", "/repo/external/SocialJax",
                   ignore=["**/.git", "**/__pycache__"])
    .add_local_dir("scripts", "/repo/scripts", ignore=["**/__pycache__"])
    .add_local_file("cleanup/train_selfreport_exploit.py",
                    "/repo/cleanup/train_selfreport_exploit.py")
)

app = modal.App("zkattr-cleanup-eval")
volume = modal.Volume.from_name("zkattr-cleanup-baseline", create_if_missing=True)

CKPT = {
    "sc": "/results/ippo_cleanup_individual_t300000000_ps0_e128_attrV2_w50_sc_seed0"
          "/checkpoints/individual",
    "recip": "/results/ippo_cleanup_common_t300000000_ps0_e128_attrV2_w50_seed0"
             "/checkpoints/individual",
}


@app.function(gpu="A100-80GB", image=image, volumes={"/results": volume}, timeout=2 * 60 * 60)
def run_eval(cli: str) -> str:
    import os
    import subprocess
    import time

    volume.reload()
    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo/external/SocialJax:/repo/src"
    cmd = ["python"] + cli.format(**CKPT).split()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=env, capture_output=True, text=True)
    out = proc.stdout
    if proc.returncode != 0:
        out += f"\n[exit {proc.returncode}] stderr tail:\n" + proc.stderr[-3000:]
    volume.commit()   # persist any --out written under /results (e.g. training CSVs)
    return out + f"\n[A100 wall: {(time.time()-t0)/60:.1f} min]"


@app.local_entrypoint()
def main(cli: str = "cleanup/train_selfreport_exploit.py --sabotage "
                    "--ckpt-dir {recip} --focal-slot 1 --episodes 32"):
    print(f"eval on A100:  {cli}\n")
    print(run_eval.remote(cli))
