#!/bin/bash
# =============================================================================
# startup.sh — project-specific startup for autoresearch
# Universal setup (Claude, tmux, uv, git) is handled by entrypoint.sh.
# This script handles project-specific deps and data only.
# =============================================================================
set -euo pipefail

REPO_NAME="${REPO_NAME:-autoresearch}"
REPO_DIR="${REPO_DIR:-/workspace/home/coder/$REPO_NAME}"
ENV_FILE="$REPO_DIR/.env"
BASHRC="${HOME}/.bashrc"

echo "[startup] $(date)"

# -----------------------------------------------------------------------------
# 1. Source .env for this session
# -----------------------------------------------------------------------------
if [ -f "$ENV_FILE" ]; then
    echo "[startup] Sourcing $ENV_FILE"
    set -a; source "$ENV_FILE"; set +a
else
    echo "[startup] NOTE: $ENV_FILE not found — create it with HF_TOKEN, WANDB_KEY, etc."
fi

# -----------------------------------------------------------------------------
# 2. Standard directories
# -----------------------------------------------------------------------------
mkdir -p ~/data/datasets ~/data/checkpoints ~/data/logs ~/data/wandb ~/data/.cache/huggingface
mkdir -p ~/scratch/tmp ~/scratch/compile

# -----------------------------------------------------------------------------
# 3. Export standard paths
# -----------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-$HOME/data/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-$HOME/data/wandb}"
export TMPDIR="${TMPDIR:-$HOME/scratch/tmp}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"
export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"

# -----------------------------------------------------------------------------
# 4. Install project dependencies
# -----------------------------------------------------------------------------
PYTHON=$(command -v python3)
echo "[startup] Using Python: $($PYTHON --version)"

echo "[startup] Installing project dependencies..."
UV=$(command -v uv)
sudo -E "$UV" pip install --system --python "$PYTHON" "$REPO_DIR"

# flash-attn is required by train.py but not in pyproject.toml (needs CUDA to build)
if ! "$PYTHON" -c "import flash_attn" 2>/dev/null; then
    echo "[startup] Installing flash-attn (may take several minutes)..."
    sudo -E "$UV" pip install --system --python "$PYTHON" setuptools
    sudo -E "$UV" pip install --system --python "$PYTHON" flash-attn --no-build-isolation
fi

# -----------------------------------------------------------------------------
# 5. DATA SETUP — download data if not already present
# -----------------------------------------------------------------------------
DATA_READY_FLAG="$HOME/data/datasets/.ready"
if [ ! -f "$DATA_READY_FLAG" ]; then
    echo "[startup] Downloading data (prepare.py)..."
    cd "$REPO_DIR"
    "$PYTHON" prepare.py
    touch "$DATA_READY_FLAG"
else
    echo "[startup] Data already present, skipping download"
fi

# -----------------------------------------------------------------------------
# 6. Convenience symlinks into repo dir
# -----------------------------------------------------------------------------
ln -sfn ~/data/checkpoints "$REPO_DIR/checkpoints" 2>/dev/null || true
ln -sfn ~/data/datasets    "$REPO_DIR/datasets"    2>/dev/null || true
ln -sfn ~/data/logs        "$REPO_DIR/logs"        2>/dev/null || true

echo "[startup] Ready — repo at $REPO_DIR"
cd "$REPO_DIR"
