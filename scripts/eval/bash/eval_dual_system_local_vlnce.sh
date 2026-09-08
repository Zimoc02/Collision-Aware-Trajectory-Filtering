#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

unset __EGL_VENDOR_LIBRARY_DIRS

python scripts/eval/eval.py \
  --config scripts/eval/configs/habitat_dual_system_local_vlnce_cfg.py
