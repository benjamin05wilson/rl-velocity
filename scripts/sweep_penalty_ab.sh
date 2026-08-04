#!/usr/bin/env bash
# A/B: does the sampling penalty inherited from generation_config.json change learning?
#
# Two conditions, three seeds each. Identical in every respect except the repetition
# penalty applied during rollout -- 1.0 (correct) vs 1.1 (the value Qwen2.5 leaks).
# The training forward pass never applies a penalty in either condition, which is the
# whole point: condition B is the on-policy assumption being violated.
#
# Seeds are matched across conditions, so seed 0 sees the same prompt order in both.
# Sequential rather than parallel because both conditions want most of the 24GB card,
# and co-residency would change the throughput each one sees.
set -uo pipefail

cd /root/rl-velocity
source env.sh

STEPS=250
EVAL_EVERY=50
EVAL_PROMPTS=200
PROMPTS=8
GROUP=8

run () {
  local penalty="$1" seed="$2" name="$3"
  if [ -d "runs/${name}" ]; then
    echo "== skip ${name} (exists)"
    return 0
  fi
  echo "== ${name}  penalty=${penalty} seed=${seed}  $(date +%H:%M:%S)"
  timeout 5400 ./.venv/bin/python -m rlv.train \
    --rollout-backend vllm \
    --steps "${STEPS}" \
    --prompts-per-step "${PROMPTS}" \
    --group-size "${GROUP}" \
    --seed "${seed}" \
    --rollout-repetition-penalty "${penalty}" \
    --eval-every "${EVAL_EVERY}" \
    --eval-prompts "${EVAL_PROMPTS}" \
    --run-name "${name}" \
    > "/root/logs_${name}.log" 2>&1
  echo "   exit=$? $(grep -c '^step' "/root/logs_${name}.log" 2>/dev/null) steps  $(date +%H:%M:%S)"
  grep '\[eval\]' "/root/logs_${name}.log" | tail -2
}

for seed in 0 1 2; do
  run 1.0 "${seed}" "ab-neutral-s${seed}"
  run 1.1 "${seed}" "ab-penalty-s${seed}"
done

echo "SWEEP_COMPLETE"
