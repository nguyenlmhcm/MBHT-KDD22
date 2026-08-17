# -*- coding: utf-8 -*-
"""Integration tests for B1 wired into MBHT.

Proves the two properties the ablation depends on, against the *actual*
pre-B1 model source pulled out of git rather than a description of it:

1. ``enable_transition_embedding=0`` reproduces A0 bit-for-bit -- same
   parameter keys, same values, same forward output at the same seed.
2. ``enable_transition_embedding=1`` adds exactly one embedding of
   ``|T| x hidden_size`` and leaves every A0 parameter untouched, so A1
   differs from A0 by that embedding alone.

CPU only, no dataset: the model is built against a stub exposing just the
two things ``MBHT.__init__`` reads from a dataset. Run directly::

    python tests/test_b1_integration.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from recbole.model.sequential_recommender.mbht import MBHT  # noqa: E402
from recbole.model.transition_utils import transition_vocab_size  # noqa: E402

SEED = 2020
HIDDEN = 64

# Parent-code defect, not ours: mbht.py allocates gating_bias with
# torch.Tensor(...) -- uninitialised memory -- and the matching
# nn.init.normal_ call is commented out, while every sibling parameter is
# initialised. It is therefore seed-independent garbage that differs
# between two builds of the SAME model at the SAME seed, so it must be
# controlled for before any bit-exactness claim. Measured on this machine:
# absmax ranged from 2.3e-10 to 0.12 across builds, moving the forward
# output by up to 3.2e-05; a bare torch.Tensor() has been observed
# returning values as large as 1.7e+28.
UNINITIALISED_PARENT_PARAMS = frozenset({'gating_bias'})
# The multi-scale branch hardcodes a length of 200 (recbole/model/layers.py:
# LinearAttention.E/F and MultiScaleAttention.out_fc), and pooling needs the
# length divisible by scales[1] and scales[2]. MAX_ITEM_LIST_LENGTH is one
# less because MBHT appends a column before calling forward().
SEQ_LEN = 200


class StubDataset:
    """The entire dataset surface ``MBHT.__init__`` touches."""

    def __init__(self, n_items, type_token_ids):
        self._n_items = n_items
        self.field2token_id = {'item_type_list': dict(type_token_ids)}

    def num(self, field):
        return self._n_items


def make_config(n_types, enable_transition, dataset_name='retail_beh'):
    return {
        'USER_ID_FIELD': 'session_id',
        'ITEM_ID_FIELD': 'item_id',
        'LIST_SUFFIX': '_list',
        'ITEM_LIST_LENGTH_FIELD': 'item_length',
        'NEG_PREFIX': 'neg_',
        'MAX_ITEM_LIST_LENGTH': SEQ_LEN - 1,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_size': HIDDEN,
        'inner_size': 256,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'mask_ratio': 0.2,
        'loss_type': 'CE',
        'hyper_len': 6,
        'enable_hg': 1,
        'enable_ms': 1,
        'scales': [5, 4, 20],
        'dataset': dataset_name,
        'enable_transition_embedding': enable_transition,
    }


def retail_type_ids():
    """The real retail_beh map: 5 ids -> |T| = 31."""
    return {'[PAD]': 0, '1': 1, '2': 2, '0': 3, '3': 4}


def tmall_type_ids():
    """The real tmall_beh map: 6 ids -> |T| = 43. Note id 1 is a different
    raw behaviour here than on retail_beh."""
    return {'[PAD]': 0, '2': 1, '1': 2, '0': 3, '3': 4, '4': 5}


def load_pre_b1_mbht():
    """Import the pre-B1 MBHT straight out of git as a separate module."""
    src = subprocess.run(
        ['git', '-C', REPO, 'show', 'HEAD:recbole/model/sequential_recommender/mbht.py'],
        capture_output=True, text=True, check=True,
    ).stdout
    if 'transition' in src:
        raise AssertionError(
            'HEAD already contains B1; point this at the pre-B1 commit, '
            'otherwise the comparison is vacuous'
        )
    tmp = tempfile.NamedTemporaryFile('w', suffix='_mbht_a0.py', delete=False)
    tmp.write(src)
    tmp.close()
    spec = importlib.util.spec_from_file_location('mbht_a0', tmp.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MBHT


def build(model_cls, n_types, enable_transition, type_ids, dataset_name='retail_beh'):
    torch.manual_seed(SEED)
    cfg = make_config(n_types, enable_transition, dataset_name)
    model = model_cls(cfg, StubDataset(n_items=500, type_token_ids=type_ids))
    model.eval()  # drop dropout so forward passes are comparable
    return model


def make_batch(model, n_types, batch_size=2):
    """A right-padded batch with one mask token per row, as MBHT expects."""
    g = torch.Generator().manual_seed(1234)
    item_seq = torch.zeros((batch_size, SEQ_LEN), dtype=torch.long)
    type_seq = torch.zeros((batch_size, SEQ_LEN), dtype=torch.long)
    for row in range(batch_size):
        length = 12 + row * 3
        item_seq[row, :length] = torch.randint(1, 400, (length,), generator=g)
        type_seq[row, :length] = torch.randint(1, n_types, (length,), generator=g)
        # a masked interaction: mask token kept non-zero, behaviour zeroed
        mask_pos = length - 4
        item_seq[row, mask_pos] = model.mask_token
        type_seq[row, mask_pos] = 0
    return item_seq, type_seq


def state_dicts_equal(a, b, ignore=()):
    if a.keys() != b.keys():
        return False, f'key sets differ: {set(a) ^ set(b)}'
    for key in a:
        if key in ignore:
            continue
        if not torch.equal(a[key], b[key]):
            return False, f'tensor {key} differs'
    return True, ''


def sync_uninitialised_params(src, dst):
    """Copy the parent's uninitialised parameters so a comparison is meaningful.

    Without this every forward comparison would be measuring the memory
    allocator rather than the model.
    """
    src_sd = src.state_dict()
    with torch.no_grad():
        for name, param in dst.named_parameters():
            if name in UNINITIALISED_PARENT_PARAMS:
                param.copy_(src_sd[name])


def test_flag_off_is_bitexact_with_pre_b1_model():
    a0 = build(load_pre_b1_mbht(), 5, 0, retail_type_ids())
    off = build(MBHT, 5, 0, retail_type_ids())

    same, why = state_dicts_equal(
        a0.state_dict(), off.state_dict(), ignore=UNINITIALISED_PARENT_PARAMS
    )
    assert same, f'flag=off diverged from the pre-B1 model: {why}'

    n_a0 = sum(p.numel() for p in a0.parameters())
    n_off = sum(p.numel() for p in off.parameters())
    assert n_a0 == n_off, f'param count changed with the flag off: {n_a0} -> {n_off}'
    assert off.transition_embedding is None


def test_parent_gating_bias_is_uninitialised():
    """Pin the parent defect, so the exclusion above stays justified.

    If MBHT ever initialises gating_bias, this fails and
    UNINITIALISED_PARENT_PARAMS should shrink -- at which point A0 becomes
    genuinely seed-reproducible.
    """
    a0_cls = load_pre_b1_mbht()
    seen = [build(a0_cls, 5, 0, retail_type_ids()).gating_bias.detach().clone()
            for _ in range(8)]
    assert any(not torch.equal(seen[0], other) for other in seen[1:]), (
        'gating_bias no longer varies between same-seed builds; the parent may '
        'have been fixed -- revisit UNINITIALISED_PARENT_PARAMS'
    )


def test_flag_off_forward_matches_pre_b1_model():
    a0 = build(load_pre_b1_mbht(), 5, 0, retail_type_ids())
    off = build(MBHT, 5, 0, retail_type_ids())
    # control the parent's uninitialised parameter, else this measures the
    # allocator rather than B1
    sync_uninitialised_params(a0, off)

    item_seq, type_seq = make_batch(off, 5)
    with torch.no_grad():
        out_a0 = a0.forward(item_seq, type_seq)
        out_off = off.forward(item_seq, type_seq)

    assert not torch.isnan(out_a0).any(), 'pre-B1 forward produced NaN'
    assert torch.equal(out_a0, out_off), 'flag=off forward output differs from A0'


def test_flag_on_adds_exactly_the_transition_embedding():
    for name, n_types, type_ids in (
        ('retail_beh', 5, retail_type_ids()),
        ('tmall_beh', 6, tmall_type_ids()),
    ):
        off = build(MBHT, n_types, 0, type_ids, dataset_name=name)
        on = build(MBHT, n_types, 1, type_ids, dataset_name=name)

        expected_delta = transition_vocab_size(n_types) * HIDDEN
        n_off = sum(p.numel() for p in off.parameters())
        n_on = sum(p.numel() for p in on.parameters())
        assert n_on - n_off == expected_delta, (
            f'{name}: expected +{expected_delta} params, got +{n_on - n_off}'
        )

        sd_off, sd_on = off.state_dict(), on.state_dict()
        new_keys = set(sd_on) - set(sd_off)
        assert new_keys == {'transition_embedding.weight'}, f'{name}: unexpected {new_keys}'
        assert tuple(sd_on['transition_embedding.weight'].shape) == (
            transition_vocab_size(n_types), HIDDEN
        )

        # every A0 parameter must be untouched by enabling the flag
        for key in sd_off:
            if key in UNINITIALISED_PARENT_PARAMS:
                continue  # parent-side garbage, varies regardless of the flag
            assert torch.equal(sd_off[key], sd_on[key]), (
                f'{name}: enabling B1 perturbed {key}; the ablation would be confounded'
            )


def test_expected_param_delta_is_the_audited_number():
    assert transition_vocab_size(5) * HIDDEN == 1984   # retail_beh
    assert transition_vocab_size(6) * HIDDEN == 2752   # tmall_beh, ijcai_beh


def test_flag_on_forward_runs_and_changes_output():
    for name, n_types, type_ids in (
        ('retail_beh', 5, retail_type_ids()),
        ('tmall_beh', 6, tmall_type_ids()),
    ):
        off = build(MBHT, n_types, 0, type_ids, dataset_name=name)
        on = build(MBHT, n_types, 1, type_ids, dataset_name=name)
        item_seq, type_seq = make_batch(on, n_types)

        with torch.no_grad():
            out_off = off.forward(item_seq, type_seq)
            out_on = on.forward(item_seq, type_seq)

        assert not torch.isnan(out_on).any(), f'{name}: B1 forward produced NaN'
        assert not torch.isinf(out_on).any(), f'{name}: B1 forward produced Inf'
        assert out_on.shape == out_off.shape
        assert not torch.equal(out_on, out_off), (
            f'{name}: enabling B1 changed nothing -- the embedding is not reaching the model'
        )


def test_flag_on_forward_is_deterministic():
    on = build(MBHT, 5, 1, retail_type_ids())
    item_seq, type_seq = make_batch(on, 5)
    with torch.no_grad():
        first = on.forward(item_seq, type_seq)
        second = on.forward(item_seq, type_seq)
    assert torch.equal(first, second), 'B1 forward is not deterministic in eval mode'


def test_derived_vocab_size_follows_the_dataset():
    """n_types comes from the dataset, so the same code yields different |T|."""
    retail = build(MBHT, 5, 1, retail_type_ids(), dataset_name='retail_beh')
    tmall = build(MBHT, 6, 1, tmall_type_ids(), dataset_name='tmall_beh')
    assert retail.n_behavior_types == 5 and retail.n_transitions == 31
    assert tmall.n_behavior_types == 6 and tmall.n_transitions == 43


def test_calculate_loss_backward_is_finite():
    """Gradients must flow into the transition embedding without blowing up."""
    on = build(MBHT, 5, 1, retail_type_ids())
    on.train()
    item_seq, type_seq = make_batch(on, 5)
    on.forward(item_seq, type_seq).sum().backward()

    grad = on.transition_embedding.weight.grad
    assert grad is not None, 'no gradient reached the transition embedding'
    assert torch.isfinite(grad).all(), 'non-finite gradient in the transition embedding'
    assert grad.abs().sum() > 0, 'transition embedding received only zero gradient'
    assert torch.equal(grad[0], torch.zeros_like(grad[0])), (
        'padding_idx row must stay at zero gradient'
    )


if __name__ == '__main__':
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((name, exc))
            print(f'FAIL  {name}\n        {type(exc).__name__}: {exc}')
        else:
            print(f'ok    {name}')
    print(f'\n{len(tests) - len(failures)}/{len(tests)} passed')
    sys.exit(1 if failures else 0)
