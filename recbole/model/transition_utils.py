# -*- coding: utf-8 -*-
"""B1: learnable behaviour-transition modelling for MBHT.

MBHT already embeds *what kind* of interaction happens at each position.
A transition token instead encodes the ordered pair
``(previous behaviour -> current behaviour)``, i.e. *how* the current
interaction was arrived at. A CART that follows a VIEW and a CART that
follows a BUY share a behaviour label but not an intent context.

Vocabulary layout, for a dataset whose behaviour-type vocabulary holds
``n_types`` entries (that is ``len(field2token_id['item_type_list'])``,
which already counts ``[PAD]`` at index 0)::

    id 0                    PAD_TRANSITION -- the position is padding
    id 1 + p * n_types + c  the transition (p -> c)

``c`` is the current type id in ``[0, n_types)``; ``p`` is the previous
type id, or ``n_types`` to denote START (the position has no valid
predecessor). Hence::

    |T| = 1 + (n_types + 1) * n_types

``n_types`` MUST be derived from the running dataset and never hardcoded.
RecBole assigns type ids with ``pd.factorize``, i.e. by order of first
appearance, so the same id denotes a *different* raw behaviour on
different datasets. Verified on the official processed benchmark:

    ===========  ==========  ==========  ==========
    internal id  retail_beh  tmall_beh   ijcai_beh
    ===========  ==========  ==========  ==========
    1            raw '1'     raw '2'     raw '0'
    2            raw '2'     raw '1'     raw '2'
    3            raw '0'     raw '0'     raw '3'
    ===========  ==========  ==========  ==========

and the vocabulary sizes differ too (retail_beh 5 -> |T| 31;
tmall_beh and ijcai_beh 6 -> |T| 43). Anything that needs a specific
behaviour (a purchase, say) must look it up by RAW token exactly as the
parent model does with ``field2token_id['item_type_list']['0']``.

Masking policy (a), locked before implementation: transitions are built
from the type sequence *after* masking. MBHT zeroes the behaviour type at
masked positions, so those positions legitimately appear as both source
and destination of a transition. Building transitions from the pre-mask
types would hand the model the behaviour of the very interaction it is
asked to predict, and any measured gain would be leakage rather than
signal.
"""

import torch

PAD_TRANSITION = 0


def transition_vocab_size(n_types):
    """Return |T| for a behaviour-type vocabulary of size ``n_types``.

    Args:
        n_types (int): number of behaviour-type ids, including ``[PAD]``.

    Returns:
        int: size of the transition vocabulary.
    """
    if not isinstance(n_types, int) or isinstance(n_types, bool):
        raise TypeError(f'n_types must be an int, got {type(n_types).__name__}')
    if n_types < 1:
        raise ValueError(f'n_types must be >= 1, got {n_types}')
    return 1 + (n_types + 1) * n_types


def build_transition_seq(type_seq, valid_mask, n_types):
    """Map a behaviour-type sequence to transition tokens.

    Pure and deterministic: draws no randomness, keeps no state, and does
    not mutate its inputs, so the same arguments always yield the same
    tensor.

    A position whose predecessor is not valid is assigned a START
    transition rather than one reaching across the gap, which keeps the
    guarantee that no transition is ever formed across padding.

    Args:
        type_seq (torch.LongTensor): ``[B, L]`` behaviour type ids, already
            masked (policy (a)).
        valid_mask (torch.BoolTensor): ``[B, L]``, True where the position
            holds a real interaction. Masked-out items stay *valid*: they
            are real positions whose behaviour is hidden. In MBHT this is
            ``item_seq > 0`` -- masked items carry a non-zero mask token --
            which is the same validity notion ``get_attention_mask`` uses.
        n_types (int): behaviour-type vocabulary size, including ``[PAD]``.

    Returns:
        torch.LongTensor: ``[B, L]`` transition ids, each in ``[0, |T|)``.
    """
    if type_seq.dim() != 2:
        raise ValueError(f'type_seq must be [B, L], got shape {tuple(type_seq.shape)}')
    if type_seq.shape != valid_mask.shape:
        raise ValueError(
            f'type_seq {tuple(type_seq.shape)} and valid_mask '
            f'{tuple(valid_mask.shape)} must have the same shape'
        )
    transition_vocab_size(n_types)  # validates n_types

    if type_seq.size(1) == 0:
        return torch.zeros_like(type_seq)

    start_idx = n_types

    # The predecessor of position k is position k-1; position 0 has none.
    prev_type = torch.cat(
        [type_seq.new_zeros((type_seq.size(0), 1)), type_seq[:, :-1]], dim=1
    )
    prev_valid = torch.cat(
        [valid_mask.new_zeros((valid_mask.size(0), 1)), valid_mask[:, :-1]], dim=1
    )
    prev = torch.where(prev_valid, prev_type, torch.full_like(prev_type, start_idx))

    transition_seq = 1 + prev * n_types + type_seq
    return torch.where(valid_mask, transition_seq, torch.zeros_like(transition_seq))


def validate_transition_inputs(type_seq, transition_seq, n_types):
    """Assert type ids and derived transition ids sit in their valid ranges.

    Out-of-range behaviour types are the dangerous case: they do not crash,
    they silently alias onto another transition's slot. This synchronises
    with the device, so call it once rather than every step.

    Raises:
        AssertionError: if any id falls outside its range.
    """
    vocab = transition_vocab_size(n_types)

    type_hi = int(type_seq.max())
    type_lo = int(type_seq.min())
    if type_lo < 0 or type_hi >= n_types:
        raise AssertionError(
            f'behaviour type ids must lie in [0, {n_types}), got '
            f'[{type_lo}, {type_hi}]. n_types is derived from '
            f"len(field2token_id['item_type_list']) -- a mismatch means it was "
            f'taken from the wrong dataset.'
        )

    tr_hi = int(transition_seq.max())
    tr_lo = int(transition_seq.min())
    if tr_lo < 0 or tr_hi >= vocab:
        raise AssertionError(
            f'transition ids must lie in [0, {vocab}), got [{tr_lo}, {tr_hi}]'
        )


def decode_transition(transition_id, n_types, id2token=None):
    """Render a transition id as a readable ``prev->cur`` label.

    Args:
        transition_id (int): id in ``[0, |T|)``.
        n_types (int): behaviour-type vocabulary size, including ``[PAD]``.
        id2token (sequence, optional): ``field2id_token['item_type_list']``,
            used to name each type by its RAW token. Without it the internal
            ids are shown, which are not comparable across datasets.

    Returns:
        str: e.g. ``"raw'2'->raw'1'"``, ``"START->raw'2'"`` or ``"<PAD>"``.
    """
    vocab = transition_vocab_size(n_types)
    if not 0 <= transition_id < vocab:
        raise ValueError(f'transition_id {transition_id} outside [0, {vocab})')
    if transition_id == PAD_TRANSITION:
        return '<PAD>'

    offset = transition_id - 1
    prev, cur = divmod(offset, n_types)

    def name(type_id):
        if id2token is None:
            return f'id{type_id}'
        token = id2token[type_id]
        return '<PAD>' if str(token) == '[PAD]' else f"raw'{token}'"

    prev_name = 'START' if prev == n_types else name(prev)
    return f'{prev_name}->{name(cur)}'


def count_transitions(transition_seq, counter=None, include_pad=False):
    """Accumulate transition-id frequencies into a ``collections.Counter``.

    Args:
        transition_seq (torch.LongTensor): ``[B, L]`` transition ids.
        counter (collections.Counter, optional): accumulator to update in
            place; a new one is created when omitted.
        include_pad (bool): count padding positions too. Off by default,
            since padding says nothing about the corpus.

    Returns:
        collections.Counter: mapping transition id -> count.
    """
    import collections

    if counter is None:
        counter = collections.Counter()
    for tau in transition_seq.reshape(-1).tolist():
        if include_pad or tau != PAD_TRANSITION:
            counter[tau] += 1
    return counter


def format_transition_table(counter, n_types, id2token=None, top=None):
    """Format transition counts as a readable table (plan checklist 15.1).

    Args:
        counter (collections.Counter): from :func:`count_transitions`.
        n_types (int): behaviour-type vocabulary size, including ``[PAD]``.
        id2token (sequence, optional): ``field2id_token['item_type_list']``,
            so rows are labelled by raw token rather than internal id.
        top (int, optional): keep only the ``top`` most frequent rows.

    Returns:
        str: the rendered table.
    """
    total = sum(counter.values())
    rows = sorted(counter.items(), key=lambda kv: -kv[1])
    if top is not None:
        rows = rows[:top]

    lines = [
        f'transition frequency table  (|T|={transition_vocab_size(n_types)}, '
        f'n_types={n_types}, distinct observed={len(counter)}, total={total})',
        f'{"id":>5}  {"transition":>22} {"count":>12} {"%":>9}',
    ]
    for tau, count in rows:
        label = decode_transition(tau, n_types, id2token)
        share = (100.0 * count / total) if total else 0.0
        lines.append(f'{tau:>5}  {label:>22} {count:>12} {share:>8.4f}%')
    return '\n'.join(lines)
