"""CPU regressions: no model downloads or GPU required."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlv.instrument import PhaseTimer, Recorder, StepAccount, StepTimer
from rlv.rollout import assemble, trim_completion
from rlv.tasks import gsm8k
from rlv.train import completion_logprobs, policy_loss, validate_temperature

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ragged_masks_and_real_pad_token():
    rb = assemble([[9, 0], [3]], [[0], [4, 0]], ['a', 'b'], 1, 2, 0, 'cpu')
    assert rb.sequences.tolist() == [[9, 0, 0, 0], [0, 3, 4, 0]]
    assert rb.attention_mask.tolist() == [[1, 1, 1, 0], [0, 1, 1, 1]]
    assert rb.completion_mask.tolist() == [[False, False, True, False], [False, False, True, True]]
    assert rb.lengths.tolist() == [[1, 2]]
    assert trim_completion([4, 0, 0], 0, 0) == [4, 0]
    assert trim_completion([4, 2, 0], [2, 5], 0) == [4, 2]
    assert trim_completion([4, 0, 0], 2, 0) == [4, 0, 0]
    assert trim_completion([4, 0, 2, 0], 2, 0) == [4, 0, 2]


def test_empty_completion_and_invalid_batch():
    rb = assemble([[1]], [[]], [''], 1, 1, 0, 'cpu')
    assert rb.lengths.item() == 0
    assert not rb.completion_mask.any()
    with pytest.raises(ValueError):
        assemble([[]], [[]], [''], 1, 1, 0, 'cpu')


@pytest.mark.parametrize('temperature', [0.5, 1.0, 2.0])
def test_scoring_shift_mask_positions_temperature(temperature):
    rb = assemble([[1, 2], [2]], [[3], [1, 3]], ['a', 'b'], 1, 2, 0, 'cpu')
    logits = torch.tensor([0., 1., 2., 3.]).repeat(2, 4, 1).requires_grad_()

    def model(seq, attention_mask, position_ids, use_cache):
        assert not use_cache
        assert torch.equal(attention_mask, rb.attention_mask)
        assert position_ids.tolist() == [[0, 1, 2, 0], [0, 0, 1, 2]]
        return SimpleNamespace(logits=logits)

    lp, mask = completion_logprobs(model, rb.sequences, rb.attention_mask, rb.completion_mask, temperature)
    expected = torch.log_softmax(torch.tensor([0., 1., 2., 3.]) / temperature, -1)
    torch.testing.assert_close(lp[mask], expected[torch.tensor([3, 1, 3])])
    lp[mask].sum().backward()
    assert logits.grad[:, -1].abs().sum() == 0


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_training_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        validate_temperature(temperature)


def test_microbatch_loss_and_gradient_invariant():
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=torch.bool)
    advantages = torch.tensor([1., -2., 0.5, 3.])
    reference = None
    for size in (1, 2, 3, 4):
        lp = torch.arange(12, dtype=torch.float).view(4, 3).requires_grad_()
        losses = [policy_loss(lp[i:i+size], mask[i:i+size], advantages[i:i+size], 4) for i in range(0, 4, size)]
        total = sum(losses)
        total.backward()
        if reference is None:
            reference = (total.detach(), lp.grad.clone())
        torch.testing.assert_close(total, reference[0])
        torch.testing.assert_close(lp.grad, reference[1])
    assert reference[1][2].abs().sum() == 0


@pytest.mark.parametrize('text,gold,correct,fmt', [
    (r'answer \boxed{1,200.0}', '1200', True, True),
    ('last number 7', '7', True, False),
    (r'7 then \boxed{oops}', '7', True, False),
    (r'\boxed{nan}', '7', False, False),
    (r'\boxed{inf}', '7', False, False),
    ('no number', None, False, False),
    (r'\boxed{2} then \boxed{3}', '3', True, True),
])
def test_grader(text, gold, correct, fmt):
    grade = gsm8k.grade(text, gold)
    assert grade.correct == correct
    assert grade.format_ok == fmt


def test_record_collision_null_and_failed_status(tmp_path, monkeypatch):
    import rlv.instrument.recorder as recorder
    monkeypatch.setattr(recorder, 'environment_fingerprint', lambda: {'fixture': True})
    with pytest.raises(RuntimeError), Recorder(tmp_path, 'one') as rec:
        rec.step(StepAccount(0))
        raise RuntimeError('test failure')
    before = (tmp_path / 'one/meta.json').read_bytes()
    with pytest.raises(FileExistsError):
        Recorder(tmp_path, 'one')
    assert (tmp_path / 'one/meta.json').read_bytes() == before
    events = [json.loads(line) for line in (tmp_path / 'one/events.jsonl').read_text(encoding="utf-8").splitlines()]
    assert all(events[0][key] is None for key in ['kl', 'entropy', 'tokens_prompt', 'wall_s', 'generated_tokens_per_s', 'mem_peak_alloc_gb', 'mem_frag_gb'])
    assert events[-1]['status'] == 'failed'
    with pytest.raises(ValueError):
        Recorder(tmp_path, '../escape')


def test_step_timer_sync_order(monkeypatch):
    import rlv.instrument.clock as clock
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append('sync'))
    ticks = iter([10., 13.])
    def tick():
        calls.append('clock')
        return next(ticks)
    monkeypatch.setattr(clock.time, 'perf_counter', tick)
    timer = StepTimer()
    timer.start()
    calls.append('work')
    assert timer.stop() == 3
    assert calls == ['sync', 'clock', 'work', 'sync', 'clock']


def test_cpu_phase_is_not_gpu_time():
    timer = PhaseTimer(enabled=False)
    with timer('cpu'):
        pass
    snap = timer.snapshot()['cpu']
    assert snap['device_s'] is None
    assert 'idle_s' not in snap


def test_equal_rewards_divergent_outputs_never_equivalent():
    compare = script('compare_backends').compare
    hf = dict(config={'model': 'fixture'}, prompt_ids=['p', 'q'], group_size=2,
              greedy_texts=['A', 'B'], greedy_tokens=[[1], [2]], sampled_rewards=[0, 1, 1, 0])
    vl = dict(hf, greedy_texts=['C', 'D'], greedy_tokens=[[3], [4]])
    result = compare(hf, vl)
    assert result['reward_delta_vllm_minus_hf'] == 0
    assert result['exact_text_fraction'] == result['exact_token_fraction'] == 0
    assert result['equivalence_established'] is False
    assert result['paired_prompt_se'] == 0
    del vl['prompt_ids']
    assert compare(hf, vl)['paired_prompt_se'] is None


def test_analysis_warmup_legacy_and_missing_telemetry(tmp_path):
    analyse = script('analyse')
    (tmp_path / 'meta.json').write_text('{}')
    event = dict(kind='step', step=0, wall_s=20, tokens_generated=200, reward_mean=0.5)
    path = tmp_path / 'events.jsonl'
    path.write_text(json.dumps(event) + '\n')
    assert analyse.summarise(tmp_path) is None
    path.write_text(json.dumps(event) + '\n' + json.dumps(dict(event, step=1, wall_s=2)) + '\n')
    row = analyse.summarise(tmp_path)
    assert row['step_s'] == 2
    assert row['timing_schema'] == 'legacy_phase_sum'
    assert row['gen_tok_s'] == 100
    assert row['device_s'] == {}
    assert 'rollout_share' not in row


def test_neutral_ratio_has_neutral_computed_diagnostic():
    diagnostic = script('measure_importance_ratio').interpretation
    result = diagnostic(torch.zeros(8))
    assert '0.0%' in result and 'p1=1.0000' in result and 'mean=1.0000' in result
    assert '37%' not in result


def test_hf_adapter_uses_attention_not_token_identity(monkeypatch):
    import sys

    from rlv.rollout import HFRollout

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(GenerationConfig=lambda **kw: kw))

    class Encoding(dict):
        def __getattr__(self, key):
            return self[key]

        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = eos_token_id = 0
        bos_token_id = 9

        def __call__(self, *args, **kwargs):
            return Encoding(input_ids=torch.tensor([[9, 0], [0, 3]]), attention_mask=torch.tensor([[1, 1], [0, 1]]))

        def decode(self, ids, **kwargs):
            return str(ids)

    class Model:
        device = 'cpu'
        training = False
        is_gradient_checkpointing = False

        def generate(self, **kw):
            assert kw['top_k'] == 0 and kw['repetition_penalty'] == 1
            assert kw['temperature'] == 0.5
            return torch.tensor([[9, 0, 4, 0, 0], [0, 3, 2, 2, 0]])

    model = Model()
    backend = HFRollout(model, Tokenizer())
    result = backend.generate(['p', 'q'], 1, 3, 0.5)
    assert result.attention_mask.tolist() == [[1, 1, 1, 1, 0], [0, 1, 1, 1, 1]]
    assert result.lengths.tolist() == [[2], [3]]
    assert result.sequences[0, 1] == 0  # a genuine prompt EOS survives
    assert model.generation_config == {'bos_token_id': 9, 'eos_token_id': 0, 'pad_token_id': 0}


def test_checked_in_replay_is_reproducible():
    import subprocess
    import sys
    analyse = script('analyse')
    rows = [analyse.summarise(ROOT / 'evidence' / name) for name in ['hf-fixed', 'vllm-fixed']]
    assert analyse.render_svg(rows) == (ROOT / 'evidence/phase-times.svg').read_text(encoding="utf-8")
    output = subprocess.check_output([sys.executable, str(ROOT / 'scripts/analyse.py'), '--runs-dir', str(ROOT / 'evidence'), 'hf-fixed', 'vllm-fixed'], text=True)
    assert output == (ROOT / 'evidence/replay.txt').read_text(encoding="utf-8")
    compare = script('compare_backends')
    data = [json.loads((ROOT / 'evidence/comparison' / f'{name}.json').read_text(encoding="utf-8")) for name in ['hf', 'vllm']]
    assert compare.compare(*data) == json.loads((ROOT / 'evidence/comparison/report.json').read_text(encoding="utf-8"))


def test_svg_writer_explicit_utf8_and_lf(tmp_path, monkeypatch):
    analyse = script('analyse')
    rows = [analyse.summarise(ROOT / 'evidence/hf-fixed')]
    original_open = Path.open
    seen = []

    def encoding_checked_open(self, mode='r', *args, **kwargs):
        if mode == 'w':
            seen.append((kwargs.get('encoding'), kwargs.get('newline')))
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', encoding_checked_open)
    output = tmp_path / 'chart.svg'
    analyse.write_svg(output, rows)
    data = output.read_bytes()
    assert seen == [('utf-8', '\n')]
    assert '—'.encode() in data
    assert b'\r\n' not in data
    assert data == analyse.render_svg(rows).encode('utf-8')


def test_allocation_budget_and_proxy_optimum():
    import itertools
    allocation = script('simulate_allocation')
    ps = [0.1, 0.5, 0.9]
    got = allocation.greedy_allocate(ps, 8, 1, 4)
    assert sum(got) == 8
    brute = max(allocation.expected_informative(ps, list(gs)) for gs in itertools.product(range(1, 5), repeat=3) if sum(gs) == 8)
    assert allocation.expected_informative(ps, got) == pytest.approx(brute)
    for budget in [0, 13]:
        with pytest.raises(ValueError):
            allocation.greedy_allocate(ps, budget, 1, 4)
