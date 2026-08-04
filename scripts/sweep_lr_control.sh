#!/usr/bin/env bash
# Positive control: can this setup learn at all?
#
# The first A/B attempt ran the correct (neutral) arm at lr=1e-6 for 185 steps and
# held-out accuracy went 0.500 -> 0.495 -> 0.485 -> 0.475 -- flat, well inside the
# n=200 standard error of ~0.035. A null A/B result against a baseline that does not
# learn says nothing: you cannot degrade a curve that is not rising.
#
# So before comparing conditions, find a learning rate where the correct
# configuration measurably improves. Only then is the comparison interpretable.
set -uo pipefail

cd /root/rl-velocity
source env.sh

run_lr () {
  local lr="$1" name="lrctl-${2}"
  if [ -d "runs/${name}" ]; then echo "== skip ${name}"; return 0; fi
  echo "== ${name}  lr=${lr}  $(date +%H:%M:%S)"
  timeout 3600 ./.venv/bin/python -m rlv.train \
    --rollout-backend vllm \
    --steps 120 \
    --prompts-per-step 8 \
    --group-size 8 \
    --lr "${lr}" \
    --seed 0 \
    --eval-every 40 \
    --eval-prompts 200 \
    --run-name "${name}" \
    > "/root/logs_${name}.log" 2>&1
  echo "   done $(date +%H:%M:%S)"
  grep '\[eval\]' "/root/logs_${name}.log"
}

run_lr 1e-6 "1e6"
run_lr 1e-5 "1e5"
run_lr 4e-5 "4e5"

echo "LR_CONTROL_COMPLETE"
