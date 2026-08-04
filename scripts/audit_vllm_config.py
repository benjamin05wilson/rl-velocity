"""The vLLM half of the sampling-config audit.

HF `generate` inherits the checkpoint's generation_config.json for any field an explicit
GenerationConfig omits (see audit_generation_config.py). This asks the same question of
vLLM, whose engine also defaults to generation_config="auto".

Three call shapes, because frameworks use all three:

  A  llm.generate(prompts)                          -- no SamplingParams at all
  B  SamplingParams(temperature=0) omitting fields  -- the OpenRLHF / common shape
  C  SamplingParams(...) with every field pinned    -- the defensive shape

Greedy throughout, so any difference is configuration rather than sampling noise.
Qwen2.5-0.5B-Instruct ships repetition_penalty=1.1, top_k=20, top_p=0.8, temperature=0.7.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTION = (
    "Natalia sold clips to 48 friends in April, then half as many in May. "
    "How many clips did she sell altogether? Reason step by step."
)


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
    )

    # generation_config defaults to "auto" -- the checkpoint's file is read.
    llm = LLM(model=MODEL, gpu_memory_utilization=0.5, max_model_len=1024, dtype="bfloat16")

    defaults = llm.llm_engine.vllm_config.model_config.get_diff_sampling_param()
    print(f"\nvLLM default_sampling_params from the checkpoint: {defaults}")

    def text(outs) -> str:
        return outs[0].outputs[0].text

    a = text(llm.generate([prompt], use_tqdm=False))
    b = text(llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=200), use_tqdm=False))
    c = text(llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=200, top_k=0, top_p=1.0, repetition_penalty=1.0),
        use_tqdm=False,
    ))
    d = text(llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=200, top_k=0, top_p=1.0, repetition_penalty=1.1),
        use_tqdm=False,
    ))

    print("\ngreedy decode under four call shapes:")
    for label, s in (("A no SamplingParams", a), ("B partial (omits penalty)", b),
                     ("C fully pinned, penalty 1.0", c), ("D fully pinned, penalty 1.1", d)):
        print(f"  {label:<30} {len(s):>4} chars")

    print("\nresult:")
    print(f"  B == C (partial matches neutral)?          {b == c}")
    print(f"  B == D (partial matches checkpoint's 1.1)? {b == d}")
    print(f"  C == D (does the penalty change output)?   {c == d}")
    print(f"  A == C (bare call matches neutral)?        {a == c}")

    print()
    if c == d:
        print("INCONCLUSIVE: the penalty did not change greedy output on this prompt.")
        return 0
    if b == c:
        print("vLLM: an explicit SamplingParams does NOT inherit the checkpoint's values.")
        print("Omitted fields take vLLM's neutral library defaults.")
    else:
        print("vLLM: an explicit SamplingParams DOES inherit checkpoint values for omitted fields.")

    print()
    print("Net effect: HF `generate` and vLLM resolve an omitted sampling field")
    print("differently -- the checkpoint's value vs a neutral default. A framework")
    print("offering both rollout backends therefore samples from two different")
    print("distributions for the same config unless every field is pinned explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
