"""Toolchain gate for Blackwell (sm_120).

Blackwell fails in a specific, quiet way: torch imports fine, `cuda.is_available()`
returns True, and small ops appear to work -- because the wheel was built without
sm_120 kernels and you are silently running JIT-compiled PTX or falling off the
tensor-core path. You find out three weeks later when throughput is a third of what
it should be.

So we check compiled arch support explicitly, and we measure achieved TFLOPS rather
than trusting that a matmul returning correct numbers means it ran well.

Exit code 0 = safe to build on. Non-zero = do not write training code yet.
"""

from __future__ import annotations

import sys
import time

FAIL: list[str] = []
WARN: list[str] = []


def check(label: str, ok: bool, detail: str = "", fatal: bool = True, on_fail: str = "") -> bool:
    mark = "PASS" if ok else ("FAIL" if fatal else "WARN")
    note = detail if ok else (on_fail or detail)
    print(f"  [{mark}] {label}" + (f" -- {note}" if note else ""))
    if not ok:
        (FAIL if fatal else WARN).append(label)
    return ok


def main() -> int:
    print("\n=== torch ===")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] import torch -- {exc}")
        return 1

    print(f"  torch {torch.__version__}, built against CUDA {torch.version.cuda}")

    if not check("cuda.is_available()", torch.cuda.is_available()):
        return 1

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"  device: {name}")
    check("compute capability is sm_120 (Blackwell)", cap == (12, 0), f"got sm_{cap[0]}{cap[1]}")

    # The load-bearing check. `get_arch_list()` is what the wheel actually shipped
    # kernels for. If sm_120 is absent, every kernel launch is JIT-from-PTX at best.
    arch_list = torch.cuda.get_arch_list()
    print(f"  compiled archs: {' '.join(arch_list)}")
    check(
        "wheel ships native sm_120 kernels",
        any(a in ("sm_120", "sm_120a") for a in arch_list),
        on_fail="no sm_120 in arch list -- running PTX JIT, expect bad perf",
    )

    print("\n=== numerics ===")
    torch.manual_seed(0)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float32)
    got = (a @ b).cpu()
    want = a.cpu() @ b.cpu()
    check("fp32 matmul matches CPU", torch.allclose(got, want, atol=1e-3))

    check("bf16 supported", torch.cuda.is_bf16_supported())

    try:
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        _ = (x @ x).float().sum().item()
        check("bf16 matmul executes", True)
    except Exception as exc:  # noqa: BLE001
        check("bf16 matmul executes", False, str(exc))

    print("\n=== attention backends ===")
    # RL rollout throughput lives or dies on the attention path.
    try:
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
        for backend, label in (
            (SDPBackend.FLASH_ATTENTION, "flash"),
            (SDPBackend.EFFICIENT_ATTENTION, "mem-efficient"),
        ):
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, q, q)
                check(f"SDPA {label} backend", True)
            except Exception as exc:  # noqa: BLE001
                check(f"SDPA {label} backend", False, str(exc)[:80], fatal=(label == "flash"))
    except Exception as exc:  # noqa: BLE001
        check("SDPA probe", False, str(exc)[:100], fatal=False)

    print("\n=== achieved throughput ===")
    # Correctness does not imply the tensor cores were used. Measure.
    n = 8192
    x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):  # warm up autotuner / clocks
        x @ y
    torch.cuda.synchronize()

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        x @ y
    torch.cuda.synchronize()  # without this we would be timing kernel *launches*
    dt = time.perf_counter() - t0

    tflops = (2 * n**3 * iters) / dt / 1e12
    print(f"  bf16 {n}x{n} matmul: {tflops:.1f} TFLOP/s")
    # A mobile Blackwell on tensor cores should clear this comfortably; landing
    # below it is the signature of a PTX-JIT or non-tensor-core path.
    check("bf16 throughput indicates tensor-core path", tflops > 80, f"{tflops:.1f} TFLOP/s", fatal=False)

    print("\n=== memory ===")
    free, total = torch.cuda.mem_get_info()
    print(f"  {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    check("at least 20 GB visible", total / 1e9 > 20, f"{total / 1e9:.1f} GB", fatal=False)

    print()
    if FAIL:
        print(f"BLOCKED -- {len(FAIL)} fatal issue(s): {', '.join(FAIL)}")
        return 1
    if WARN:
        print(f"OK with {len(WARN)} warning(s): {', '.join(WARN)}")
    else:
        print("OK -- toolchain is sound, safe to build on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
