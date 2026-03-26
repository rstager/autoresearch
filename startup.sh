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
# 7. Install system utilities
# -----------------------------------------------------------------------------
echo "[startup] Installing system utilities..."
apt-get update -qq && apt-get install -y -qq tmux vim

# -----------------------------------------------------------------------------
# 8. Configure git credentials from GITHUB_TOKEN env var
# -----------------------------------------------------------------------------
if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "[startup] Configuring git credentials from GITHUB_TOKEN"
    git config --global credential.helper store
    echo "https://x-token:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
else
    echo "[startup] NOTE: GITHUB_TOKEN not set — git push will require manual auth"
fi

# -----------------------------------------------------------------------------
# 9. Configure git user identity
# -----------------------------------------------------------------------------
git config --global user.name "${GIT_USER_NAME:-Roger}"
git config --global user.email "${GIT_USER_EMAIL:-rkstager@gmail.com}"

# -----------------------------------------------------------------------------
# 10. Install uv if not already installed
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
# 11. Install project dependencies into system Python
# -----------------------------------------------------------------------------
echo "[startup] Installing project dependencies..."
uv pip install --system --python python3.11 \
    $(python3.11 -c "import tomllib; deps=tomllib.load(open('$REPO_DIR/pyproject.toml','rb'))['project']['dependencies']; print(' '.join(deps))")

# flash-attn is required by train.py but not in pyproject.toml (needs CUDA to build)
if ! python3.11 -c "import flash_attn" 2>/dev/null; then
    echo "[startup] Installing flash-attn (may take several minutes)..."
    uv pip install --system --python python3.11 flash-attn
fi

# -----------------------------------------------------------------------------
# 12. Restore user configs (tmux, Claude Code state)
# -----------------------------------------------------------------------------
# tmux config
cat > /root/.tmux.conf << 'EOF'
set -g mouse on
set -g default-terminal "xterm-256color"
EOF

# Restore Claude Code memory and settings from workspace backup
CLAUDE_BACKUP="$REPO_DIR/.claude-state"
CLAUDE_PROJECT="/root/.claude/projects/-workspace-autoresearch"
if [ -d "$CLAUDE_BACKUP/memory" ]; then
    echo "[startup] Restoring Claude Code memory from backup"
    mkdir -p "$CLAUDE_PROJECT/memory"
    cp -a "$CLAUDE_BACKUP/memory/"* "$CLAUDE_PROJECT/memory/" 2>/dev/null || true
fi
if [ -f "$CLAUDE_BACKUP/settings.json" ]; then
    mkdir -p /root/.claude
    cp "$CLAUDE_BACKUP/settings.json" /root/.claude/settings.json
fi

# -----------------------------------------------------------------------------
# 13. DATA SETUP — download data if not already present
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
# 14. Convenience symlinks into repo dir
# -----------------------------------------------------------------------------
ln -sfn /data/checkpoints "$REPO_DIR/checkpoints" 2>/dev/null || true
ln -sfn /data/datasets    "$REPO_DIR/datasets"    2>/dev/null || true
ln -sfn /data/logs        "$REPO_DIR/logs"        2>/dev/null || true

# -----------------------------------------------------------------------------
# 15. Install Claude Code if not already installed
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
# 16. Background sync: periodically back up Claude state to /workspace
# -----------------------------------------------------------------------------
(while true; do
    sleep 300
    mkdir -p "$CLAUDE_BACKUP/memory"
    cp -a "$CLAUDE_PROJECT/memory/"* "$CLAUDE_BACKUP/memory/" 2>/dev/null || true
    [ -f /root/.claude/settings.json ] && cp /root/.claude/settings.json "$CLAUDE_BACKUP/settings.json"
done) &

# -----------------------------------------------------------------------------
# 17. Ready — keep alive for interactive SSH use
#    To auto-start training: replace with: exec uv run train.py
# -----------------------------------------------------------------------------
echo "[startup] Ready — repo at $REPO_DIR"
cd "$REPO_DIR"
tail -f /dev/null
