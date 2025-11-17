#!/usr/bin/env bash
set -euo pipefail

BENCH_LOC="./data/benchmark.json"

# load environment variables (export -> load .env -> re-import)

set -a
. .env
set +a

# sanity check
echo "Env name is $AUTO_ENV_NAME"


# conda check (runnable in this shell)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    . "$HOME/miniconda/etc/profile.d/conda.sh"
  elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    . "/opt/conda/etc/profile.d/conda.sh"
  fi
fi


echo "[Step 1/1] Downloading MIRAGE benchmark data as 'benchmark.json' (saving to ./data) ... "

wget -qO "$BENCH_LOC" https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/refs/heads/main/benchmark.json

echo "DONE."
