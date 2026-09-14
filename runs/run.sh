#!/usr/bin/env bash
# OutMind Unified One-Command Runner
# Usage:
#   ./runs/run.sh [nano|small|medium|large] [--mode quick|full]
set -e

PRESET=${1:-nano}
shift || true

echo "================================================="
echo "   OutMind 1-Command Pipeline Launcher"
echo "   Preset: $PRESET"
echo "================================================="

case "$PRESET" in
  nano)
    python runs/run_nano.py "$@"
    ;;
  small)
    python runs/run_small.py "$@"
    ;;
  medium)
    python runs/run_medium.py "$@"
    ;;
  large)
    python runs/run_large.py "$@"
    ;;
  *)
    python runs/pipeline.py --preset "$PRESET" "$@"
    ;;
esac
