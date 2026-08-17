# -*- coding: utf-8 -*-
"""Unit tests for B1 transition construction (recbole/model/transition_utils.py).

Runs on CPU with no dataset and no GPU. Every case is exercised against
both benchmark vocabulary sizes, because they genuinely differ:

    retail_beh  n_types = 5  ->  |T| = 31
    tmall_beh   n_types = 6  ->  |T| = 43

Run directly::

    python tests/test_transition_utils.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole.model.transition_utils import (  # noqa: E402
    PAD_TRANSITION,
    build_transition_seq,
    count_transitions,
    decode_transition,
    format_transition_table,
    transition_vocab_size,
    validate_transition_inputs,
)

# The two vocabulary sizes present in the official processed benchmark.
RETAIL_N_TYPES = 5
TMALL_N_TYPES = 6
BOTH = (('retail_beh', RETAIL_N_TYPES), ('tmall_beh', TMALL_N_TYPES))


def tau(prev, cur, n_types):
    """Expected transition id, written out independently of the implementation."""
    return 1 + prev * n_types + cur


def test_vocab_size_matches_audited_numbers():
    assert transition_vocab_size(RETAIL_N_TYPES) == 31
    assert transition_vocab_size(TMALL_N_TYPES) == 43
    # formula holds generally
    for n in range(1, 12):
        assert transition_vocab_size(n) == 1 + (n + 1) * n


def test_vocab_size_rejects_bad_input():
    for bad in (0, -1):
        try:
            transition_vocab_size(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for n_types={bad}')
    for bad in (5.0, '5', True):
        try:
            transition_vocab_size(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f'expected TypeError for n_types={bad!r}')


def test_start_at_first_valid_event():
    for _, n in BOTH:
        types = torch.tensor([[2, 1, 3]])
        valid = torch.ones_like(types, dtype=torch.bool)
        out = build_transition_seq(types, valid, n)
        expected = torch.tensor([[tau(n, 2, n), tau(2, 1, n), tau(1, 3, n)]])
        assert torch.equal(out, expected), f'n_types={n}: {out} != {expected}'


def test_padding_maps_to_pad_transition():
    for _, n in BOTH:
        types = torch.tensor([[2, 1, 0, 0]])
        valid = torch.tensor([[True, True, False, False]])
        out = build_transition_seq(types, valid, n)
        assert out[0, 2].item() == PAD_TRANSITION
        assert out[0, 3].item() == PAD_TRANSITION
        # the valid prefix is unaffected by the trailing padding
        assert out[0, 0].item() == tau(n, 2, n)
        assert out[0, 1].item() == tau(2, 1, n)


def test_no_transition_across_padding():
    """A position whose predecessor is invalid gets START, never a bridge."""
    for _, n in BOTH:
        # a gap sits between two valid runs
        types = torch.tensor([[2, 0, 3]])
        valid = torch.tensor([[True, False, True]])
        out = build_transition_seq(types, valid, n)
        assert out[0, 1].item() == PAD_TRANSITION
        # position 2 must NOT encode 2->3, which would span the gap
        assert out[0, 2].item() == tau(n, 3, n), 'expected START, got a cross-pad transition'
        assert out[0, 2].item() != tau(2, 3, n)


def test_single_event_sequence():
    for _, n in BOTH:
        types = torch.tensor([[4 if n > 4 else 1]])
        valid = torch.tensor([[True]])
        out = build_transition_seq(types, valid, n)
        assert out.shape == (1, 1)
        assert out[0, 0].item() == tau(n, types[0, 0].item(), n)


def test_repeated_behavior():
    for _, n in BOTH:
        types = torch.tensor([[2, 2, 2]])
        valid = torch.ones_like(types, dtype=torch.bool)
        out = build_transition_seq(types, valid, n)
        assert out[0, 0].item() == tau(n, 2, n)
        assert out[0, 1].item() == tau(2, 2, n)
        assert out[0, 2].item() == tau(2, 2, n)


def test_masked_position_uses_post_mask_type():
    """Policy (a): a masked position carries type 0 and is still valid."""
    for _, n in BOTH:
        types = torch.tensor([[2, 0, 1]])  # position 1 masked -> type 0
        valid = torch.ones_like(types, dtype=torch.bool)
        out = build_transition_seq(types, valid, n)
        assert out[0, 0].item() == tau(n, 2, n)
        assert out[0, 1].item() == tau(2, 0, n), 'masked position must be a real (prev->0)'
        assert out[0, 2].item() == tau(0, 1, n), 'position after a mask must read (0->cur)'
        assert out[0, 1].item() != PAD_TRANSITION, 'a masked position is not padding'


def test_policy_a_differs_from_pre_mask_types():
    """Guard against regressing to the leaking variant.

    Feeding the pre-mask types must produce different tokens; if it did
    not, policy (a) would be unobservable and leakage undetectable.
    """
    for _, n in BOTH:
        pre_mask = torch.tensor([[2, 3, 1]])   # true behaviour at position 1 is 3
        post_mask = torch.tensor([[2, 0, 1]])  # masked -> 0
        valid = torch.ones_like(pre_mask, dtype=torch.bool)
        leaky = build_transition_seq(pre_mask, valid, n)
        correct = build_transition_seq(post_mask, valid, n)
        assert not torch.equal(leaky, correct)
        assert correct[0, 1].item() == tau(2, 0, n)
        assert leaky[0, 1].item() == tau(2, 3, n)


def test_truncation():
    """Truncating first, then building, must re-START at the kept head."""
    for _, n in BOTH:
        full = torch.tensor([[2, 1, 3, 2, 1]])
        valid_full = torch.ones_like(full, dtype=torch.bool)
        out_full = build_transition_seq(full, valid_full, n)

        keep = 3  # keep the last 3 positions, as truncation to max_len would
        truncated = full[:, -keep:]
        out_truncated = build_transition_seq(
            truncated, torch.ones_like(truncated, dtype=torch.bool), n
        )

        assert out_truncated[0, 0].item() == tau(n, 3, n), 'head of a truncated seq must be START'
        assert out_full[0, 2].item() == tau(1, 3, n), 'same position in the full seq is not START'
        # the remaining positions agree, since their predecessors survive
        assert torch.equal(out_truncated[0, 1:], out_full[0, -keep + 1:])


def test_batch_rows_are_independent():
    for _, n in BOTH:
        types = torch.tensor([[2, 1, 0], [3, 3, 3]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        out = build_transition_seq(types, valid, n)
        assert torch.equal(out[0], torch.tensor([tau(n, 2, n), tau(2, 1, n), PAD_TRANSITION]))
        assert torch.equal(out[1], torch.tensor([tau(n, 3, n), tau(3, 3, n), tau(3, 3, n)]))


def test_all_ids_in_range_over_every_pair():
    """Exhaustive: every (prev, cur) pair must land inside [0, |T|)."""
    for _, n in BOTH:
        vocab = transition_vocab_size(n)
        types = torch.arange(n).repeat(n, 1)                 # [n, n]
        valid = torch.ones_like(types, dtype=torch.bool)
        out = build_transition_seq(types, valid, n)
        assert int(out.min()) >= 0
        assert int(out.max()) < vocab, f'n_types={n}: max {int(out.max())} >= |T|={vocab}'
        validate_transition_inputs(types, out, n)


def test_max_id_is_reachable_and_tight():
    """|T| is not loose: the largest id is exactly |T| - 1."""
    for _, n in BOTH:
        types = torch.tensor([[n - 1]])
        valid = torch.tensor([[True]])
        out = build_transition_seq(types, valid, n)   # START -> last type
        assert out[0, 0].item() == transition_vocab_size(n) - 1


def test_validate_rejects_out_of_range_types():
    for _, n in BOTH:
        bad_types = torch.tensor([[n]])   # one past the vocabulary
        valid = torch.tensor([[True]])
        out = build_transition_seq(bad_types, valid, n)
        try:
            validate_transition_inputs(bad_types, out, n)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'n_types={n}: out-of-range type was not caught')


def test_determinism_same_input_same_output():
    for _, n in BOTH:
        types = torch.tensor([[2, 1, 0, 3, 2], [1, 1, 2, 0, 0]])
        valid = torch.tensor([
            [True, True, True, True, False],
            [True, True, True, False, False],
        ])
        first = build_transition_seq(types, valid, n)
        for _ in range(5):
            again = build_transition_seq(types, valid, n)
            assert torch.equal(first, again), f'n_types={n}: output changed between calls'


def test_inputs_are_not_mutated():
    for _, n in BOTH:
        types = torch.tensor([[2, 1, 0]])
        valid = torch.tensor([[True, True, False]])
        types_before = types.clone()
        valid_before = valid.clone()
        build_transition_seq(types, valid, n)
        assert torch.equal(types, types_before), 'type_seq was mutated'
        assert torch.equal(valid, valid_before), 'valid_mask was mutated'


def test_shape_validation():
    types = torch.tensor([[2, 1]])
    try:
        build_transition_seq(types, torch.tensor([[True]]), RETAIL_N_TYPES)
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched shapes were not rejected')

    try:
        build_transition_seq(torch.tensor([2, 1]), torch.tensor([True, True]), RETAIL_N_TYPES)
    except ValueError:
        pass
    else:
        raise AssertionError('1-D input was not rejected')


def test_empty_sequence_length():
    for _, n in BOTH:
        types = torch.zeros((2, 0), dtype=torch.long)
        valid = torch.zeros((2, 0), dtype=torch.bool)
        out = build_transition_seq(types, valid, n)
        assert out.shape == (2, 0)


def test_decode_transition_labels():
    n = RETAIL_N_TYPES
    id2token = ['[PAD]', '1', '2', '0', '3']   # real retail_beh mapping
    assert decode_transition(PAD_TRANSITION, n, id2token) == '<PAD>'
    assert decode_transition(tau(n, 2, n), n, id2token) == "START->raw'2'"
    assert decode_transition(tau(2, 3, n), n, id2token) == "raw'2'->raw'0'"
    # a masked destination shows up as PAD on the current side
    assert decode_transition(tau(2, 0, n), n, id2token) == "raw'2'-><PAD>"
    # without id2token, internal ids are shown rather than guessed tokens
    assert decode_transition(tau(2, 3, n), n) == 'id2->id3'

    try:
        decode_transition(transition_vocab_size(n), n, id2token)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range transition id was not rejected')


def test_count_and_format_table():
    n = RETAIL_N_TYPES
    id2token = ['[PAD]', '1', '2', '0', '3']
    types = torch.tensor([[2, 2, 1, 0], [2, 1, 0, 0]])
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, True, False],
    ])
    out = build_transition_seq(types, valid, n)

    counter = count_transitions(out)
    assert PAD_TRANSITION not in counter, 'padding must be excluded by default'
    assert counter[tau(n, 2, n)] == 2      # both rows start START->'2'
    assert counter[tau(2, 2, n)] == 1
    assert counter[tau(2, 1, n)] == 2
    assert counter[tau(1, 0, n)] == 1
    assert sum(counter.values()) == 6      # 3 valid positions x 2 rows

    with_pad = count_transitions(out, include_pad=True)
    assert with_pad[PAD_TRANSITION] == 2

    # accumulating into an existing counter doubles the counts
    again = count_transitions(out, counter=count_transitions(out))
    assert again[tau(n, 2, n)] == 4

    table = format_transition_table(counter, n, id2token)
    assert "START->raw'2'" in table
    assert '|T|=31' in table
    assert len(format_transition_table(counter, n, id2token, top=2).splitlines()) == 4


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
