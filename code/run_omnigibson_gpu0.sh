#!/usr/bin/env bash
set -euo pipefail

DELTASG_GPU=0 exec "$(dirname "$0")/run_omnigibson_single_gpu.sh" "$@"
