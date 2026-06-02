#!/bin/bash
# install.sh — automates the RECOVAR SeisComP integration setup.
# Run from inside the recovar repository root.
# Usage: bash seiscomp_integration/install.sh [--record-stream <url>]
#
# What this script does:
#   1. Installs system packages (requires sudo)
#   2. Creates the Python venv and installs dependencies
#   3. Installs the pick filter into SeisComP
#   4. Writes the pick filter config and global.cfg
#   5. Updates ~/.bashrc with the required environment variables
#
# What you still need to do manually BEFORE running this script:
#   - Download and extract SeisComP to ~/seiscomp
#   - Run: seiscomp setup  (interactive wizard)
#   - Run: seiscomp start scmaster

set -e

# ── Configurable paths ────────────────────────────────────────────────────────
SEISCOMP_ROOT="${SEISCOMP_ROOT:-$HOME/seiscomp}"
VENV="$HOME/recovar-seiscomp"
REPO="$(cd "$(dirname "$0")/.." && pwd)"   # repo root, works from any directory
RECORD_STREAM="slink://localhost:18000"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --record-stream) RECORD_STREAM="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== RECOVAR SeisComP Integration Installer ==="
echo "  SEISCOMP_ROOT : $SEISCOMP_ROOT"
echo "  venv          : $VENV"
echo "  repo          : $REPO"
echo "  recordStream  : $RECORD_STREAM"
echo ""

# ── Step 1: system packages ───────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get install -y libboost-program-options1.74.0 mariadb-server mariadb-client
sudo systemctl start mariadb
sudo systemctl enable mariadb
echo "      system packages OK"

# ── Step 2: Python venv ───────────────────────────────────────────────────────
echo "[2/5] Creating Python 3.10 venv at $VENV..."
python3.10 -m venv "$VENV"
"$VENV/bin/pip" install --quiet tensorflow==2.14.0 numpy==1.26.0 scipy obspy
echo "      venv OK"

# ── Step 3: install pick filter and batch test binaries ──────────────────────
echo "[3/5] Installing pick filter and batch test..."
cp "$REPO/seiscomp_integration/recovar_pick_filter.py" "$SEISCOMP_ROOT/bin/recovar_pick_filter"
sed -i "1s|.*|#!$VENV/bin/python3|" "$SEISCOMP_ROOT/bin/recovar_pick_filter"
chmod +x "$SEISCOMP_ROOT/bin/recovar_pick_filter"
echo "      pick filter installed at $SEISCOMP_ROOT/bin/recovar_pick_filter"

cp "$REPO/seiscomp_integration/batch_score_test.py" "$SEISCOMP_ROOT/bin/recovar_batch_test"
sed -i "1s|.*|#!$VENV/bin/python3|" "$SEISCOMP_ROOT/bin/recovar_batch_test"
chmod +x "$SEISCOMP_ROOT/bin/recovar_batch_test"
echo "      batch test installed at $SEISCOMP_ROOT/bin/recovar_batch_test"

cp "$REPO/seiscomp_integration/recovar_pick_filter.py.init" "$SEISCOMP_ROOT/etc/init/recovar_pick_filter.py"
echo "      init descriptor installed at $SEISCOMP_ROOT/etc/init/recovar_pick_filter.py"

seiscomp enable recovar_pick_filter
echo "      recovar_pick_filter enabled"

# ── Step 4: write config files ────────────────────────────────────────────────
echo "[4/5] Writing config files..."

cat > "$SEISCOMP_ROOT/etc/recovar_pick_filter.cfg" << EOF
recovar.modelPath = $REPO/models/representation_cross_covariances.h5
recordStream      = $RECORD_STREAM
messaging.hostname = localhost
agencyID = TEST
EOF
echo "      wrote $SEISCOMP_ROOT/etc/recovar_pick_filter.cfg"

mkdir -p "$HOME/.seiscomp"
if [ ! -f "$HOME/.seiscomp/global.cfg" ]; then
    cat > "$HOME/.seiscomp/global.cfg" << EOF
agencyID = TEST
organization = TEST
core.plugins = dbmysql
EOF
    echo "      wrote $HOME/.seiscomp/global.cfg"
else
    # Ensure core.plugins is present without overwriting user settings
    if ! grep -q "core.plugins" "$HOME/.seiscomp/global.cfg"; then
        echo "core.plugins = dbmysql" >> "$HOME/.seiscomp/global.cfg"
        echo "      appended core.plugins to existing $HOME/.seiscomp/global.cfg"
    else
        echo "      $HOME/.seiscomp/global.cfg already exists, skipped"
    fi
fi

# ── Step 5: update ~/.bashrc ──────────────────────────────────────────────────
echo "[5/5] Updating ~/.bashrc..."
MARKER="# SeisComP + RECOVAR environment"
if grep -q "$MARKER" "$HOME/.bashrc"; then
    echo "      ~/.bashrc already contains SeisComP environment, skipped"
else
    cat >> "$HOME/.bashrc" << EOF

$MARKER
export SEISCOMP_ROOT=$SEISCOMP_ROOT
export PATH=/usr/bin:\$SEISCOMP_ROOT/bin:\$PATH
export LD_LIBRARY_PATH=\$SEISCOMP_ROOT/lib
export PYTHONPATH=\$SEISCOMP_ROOT/lib/python:$REPO:$REPO/seiscomp_integration
EOF
    echo "      appended to ~/.bashrc"
fi

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying imports ==="
export LD_LIBRARY_PATH="$SEISCOMP_ROOT/lib"
export PYTHONPATH="$SEISCOMP_ROOT/lib/python:$REPO:$REPO/seiscomp_integration"

"$VENV/bin/python3" -c "
import seiscomp.client, seiscomp.datamodel, seiscomp.io
import tensorflow, recovar, recovar_scorer
print('All imports OK — tensorflow', tensorflow.__version__)
" 2>/dev/null && echo "Verification passed." || echo "Verification FAILED — check the output above."

echo ""
echo "=== Done ==="
echo "Run 'seiscomp start scmaster' if not already running, then:"
echo "  source ~/.bashrc"
echo "  $VENV/bin/python3 $SEISCOMP_ROOT/bin/recovar_pick_filter"
