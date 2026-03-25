#!/bin/bash
# =============================================================================
# startup.sh — Lives in the repo at /workspace/autoresearch/startup.sh
# Called by entrypoint.sh after clone/pull.
# =============================================================================
set -euo pipefail

REPO_NAME="${REPO_NAME:-autoresearch}"
REPO_DIR="/workspace/$REPO_NAME"
ENV_FILE="$REPO_DIR/.env"
BASHRC="/root/.bashrc"

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
# 2. Patch .bashrc so .env is set in every shell
# -----------------------------------------------------------------------------
BASHRC_MARKER="# cloud: auto-source project .env"
if ! grep -qF "$BASHRC_MARKER" "$BASHRC" 2>/dev/null; then
    echo "[startup] Patching $BASHRC"
    cat >> "$BASHRC" << BASHEOF

$BASHRC_MARKER
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi
BASHEOF
fi

# -----------------------------------------------------------------------------
# 3. GPU check
# -----------------------------------------------------------------------------
echo "[startup] GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || echo "[startup] WARNING: nvidia-smi failed"

# -----------------------------------------------------------------------------
# 4. Volume / disk check
# -----------------------------------------------------------------------------
for vol in /workspace /data /scratch; do
    if [ -d "$vol" ]; then
        echo "[startup] $vol: $(df -h "$vol" | tail -1 | awk '{print $4}') free"
    else
        echo "[startup] WARNING: $vol not available"
    fi
done

# -----------------------------------------------------------------------------
# 5. Standard directories
# -----------------------------------------------------------------------------
mkdir -p /data/datasets /data/checkpoints /data/logs /data/wandb /data/.cache/huggingface
mkdir -p /scratch/tmp /scratch/compile

# -----------------------------------------------------------------------------
# 6. Export standard paths
# -----------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/data/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-/data/wandb}"
export TMPDIR="${TMPDIR:-/scratch/tmp}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

# -----------------------------------------------------------------------------
# 7. Install uv if not already installed
# -----------------------------------------------------------------------------
UV_BASHRC_MARKER="# cloud: uv PATH"
if ! command -v uv &>/dev/null; then
    echo "[startup] Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
fi
if ! grep -qF "$UV_BASHRC_MARKER" "$BASHRC" 2>/dev/null; then
    echo "[startup] Adding uv to PATH in $BASHRC"
    cat >> "$BASHRC" << BASHEOF

$UV_BASHRC_MARKER
export PATH="\$HOME/.local/bin:\$PATH"
BASHEOF
fi
export PATH="$HOME/.local/bin:$PATH"
if command -v uv &>/dev/null; then
    echo "[startup] uv ready: $(uv --version)"
else
    echo "[startup] WARNING: uv binary not found after install"
fi

# -----------------------------------------------------------------------------
# 8. Install project dependencies into system Python
# -----------------------------------------------------------------------------
echo "[startup] Installing project dependencies..."
uv pip install --system --python python3.11 \
    $(python3.11 -c "import tomllib; deps=tomllib.load(open('$REPO_DIR/pyproject.toml','rb'))['project']['dependencies']; print(' '.join(deps))")

# -----------------------------------------------------------------------------
# 9. DATA SETUP — download data if not already present
# -----------------------------------------------------------------------------
DATA_READY_FLAG="/data/datasets/.ready"
if [ ! -f "$DATA_READY_FLAG" ]; then
    echo "[startup] Downloading data (prepare.py)..."
    cd "$REPO_DIR"
    python3.11 prepare.py
    touch "$DATA_READY_FLAG"
else
    echo "[startup] Data already present, skipping download"
fi

# -----------------------------------------------------------------------------
# 10. Convenience symlinks into repo dir
# -----------------------------------------------------------------------------
ln -sfn /data/checkpoints "$REPO_DIR/checkpoints" 2>/dev/null || true
ln -sfn /data/datasets    "$REPO_DIR/datasets"    2>/dev/null || true
ln -sfn /data/logs        "$REPO_DIR/logs"        2>/dev/null || true

# -----------------------------------------------------------------------------
# 11. Install Claude Code if not already installed
# -----------------------------------------------------------------------------
CLAUDE_BASHRC_MARKER="# cloud: claude-code PATH"
if ! command -v claude &>/dev/null; then
    echo "[startup] Installing Claude Code..."
    curl -fsSL https://claude.ai/install.sh | bash
fi
if ! grep -qF "$CLAUDE_BASHRC_MARKER" "$BASHRC" 2>/dev/null; then
    echo "[startup] Adding Claude Code to PATH in $BASHRC"
    cat >> "$BASHRC" << BASHEOF

$CLAUDE_BASHRC_MARKER
export PATH="\$HOME/.claude/bin:\$PATH"
BASHEOF
fi
export PATH="$HOME/.claude/bin:$PATH"
if command -v claude &>/dev/null; then
    echo "[startup] Claude Code ready: $(claude --version 2>/dev/null || true)"
else
    echo "[startup] WARNING: claude binary not found after install"
fi

# -----------------------------------------------------------------------------
# 12. Ready — keep alive for interactive SSH use
#    To auto-start training: replace with: exec uv run train.py
# -----------------------------------------------------------------------------
echo "[startup] Ready — repo at $REPO_DIR"
cd "$REPO_DIR"
tail -f /dev/null
