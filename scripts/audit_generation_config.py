"""Does passing an explicit GenerationConfig actually override the model's own?

This decides whether the sampling-config problem is a bug in one RL framework or a
property of transformers that every framework has to defend against.

The test is behavioural rather than introspective. Greedy decoding is deterministic, so
if a config that *omits* repetition_penalty produces the model's-penalty output rather
than the neutral one, the model's generation_config.json leaked in.

  A  GenerationConfig(do_sample=False)                          <- omits the field
  B  GenerationConfig(do_sample=False, repetition_penalty=1.0)  <- explicit neutral
  C  GenerationConfig(do_sample=False, repetition_penalty=1.1)  <- the model's value

A == B  -> explicit config wins, frameworks are safe by default
A == C  -> the model's config leaks through, and omitting the field is a live bug
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTION = (
    "Natalia sold clips to 48 friends in April, then half as many in May. "
    "How many clips did she sell altogether? Reason step by step."
)


def main() -> int:
    import transformers

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    print(f"transformers {transformers.__version__}")
    gc_model = model.generation_config
    print(f"model generation_config: repetition_penalty={gc_model.repetition_penalty} "
          f"top_k={gc_model.top_k} top_p={gc_model.top_p} temperature={gc_model.temperature}")
    print(f"GenerationConfig() class default: repetition_penalty={GenerationConfig().repetition_penalty}")

    text = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
    )
    enc = tok([text], return_tensors="pt").to(model.device)

    def gen(label: str, **kw) -> str:
        cfg = GenerationConfig(do_sample=False, max_new_tokens=200, **kw)
        with torch.no_grad():
            out = model.generate(**enc, generation_config=cfg, pad_token_id=tok.pad_token_id)
        s = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {label:<34} {len(s):>4} chars")
        return s

    print("\ngreedy decode under three configs:")
    a = gen("A omits repetition_penalty")
    b = gen("B explicit 1.0 (neutral)", repetition_penalty=1.0)
    c = gen("C explicit 1.1 (model's value)", repetition_penalty=1.1)

    print("\nresult:")
    print(f"  A == B (neutral)?      {a == b}")
    print(f"  A == C (model's 1.1)?  {a == c}")
    print(f"  B == C?                {b == c}   <- if True the penalty had no effect here")

    print()
    if a == c and b != c:
        print("LEAK CONFIRMED: omitting a field falls back to the model's generation_config.")
        print("Any framework that builds GenerationConfig without naming every sampling")
        print("field inherits the checkpoint's values into its RL rollouts.")
    elif a == b and b != c:
        print("NO LEAK: an explicit GenerationConfig wins; omitted fields take library")
        print("defaults, not the checkpoint's.")
    else:
        print("INCONCLUSIVE: the penalty did not change the output on this prompt.")
        print("Re-run with a prompt whose greedy decode repeats tokens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
