#!/usr/bin/env bash
# Run a SocialJax MAPPO baseline from the zkAttribution repo root.
#
# Usage:
#   scripts/run_baseline.sh <env> [hydra overrides...]
#
#   env: "cleanup" or "harvest_common"
#
# Examples:
#   scripts/run_baseline.sh cleanup
#   scripts/run_baseline.sh cleanup TOTAL_TIMESTEPS=10000
#   scripts/run_baseline.sh harvest_common SEED=42 NUM_SEEDS=5

set -euo pipefail

ENV="${1:-cleanup}"
shift || true

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

case "$ENV" in
  cleanup)
    SCRIPT="external/SocialJax/algorithms/MAPPO/mappo_cnn_cleanup.py"
    ;;
  harvest_common)
    SCRIPT="external/SocialJax/algorithms/MAPPO/mappo_cnn_harvest_common.py"
    ;;
  *)
    echo "Unknown env: $ENV" >&2
    echo "Available: cleanup, harvest_common" >&2
    exit 1
    ;;
esac

export PYTHONPATH="external/SocialJax:${PYTHONPATH:-}"
WANDB_MODE="${WANDB_MODE:-offline}"

# WANDB_MODE is read from the hydra config (config["WANDB_MODE"]), not the env var.
# Inject it as the first hydra override so the env var still controls behavior.
uv run python "$SCRIPT" "WANDB_MODE=${WANDB_MODE}" "$@"
