"""Tokenization + windowing core.

The pure, side-effect-free half of the dataset: turn one region set into a list of
fixed-size token windows. Generalizes the offline atacformer pretokenization
(``benchmarking/bedbase_bulk/tokenization/pretokenize.py``):

1. tokenize the region set against the universe,
2. drop ``unk`` tokens (regions that hit no universe interval),
3. window the remaining ids into ``context_size`` chunks by *sample-and-remove*,
4. skip pathologically large files (more than ``max_windows`` windows' worth).

No source, no dataset, no I/O -- just ids in, windows out -- so it is trivially
unit-testable and reusable by any model.
"""

import random
from typing import List, Optional


def sample_and_remove(
    ids: List[int], context_size: int, rng: Optional[random.Random] = None
) -> List[List[int]]:
    """Window a flat token list into ``context_size`` chunks by sampling without replacement.

    Repeatedly draw ``context_size`` tokens uniformly at random (without
    replacement) and remove them, until fewer than ``context_size`` remain; the
    final partial window is kept if non-empty. This randomizes which regions
    co-occur in a window across the whole file, which is what we want for a
    set-based (order-invariant) model -- it removes the positional bias that plain
    front-to-back chunking would bake in. Implemented as a shuffle-then-chunk, which
    is distributionally identical to iterated sample-and-remove but O(n) instead of
    O(n^2).

    Args:
        ids: flat list of token ids for one region set.
        context_size: number of tokens per window.
        rng: optional ``random.Random`` for reproducible shuffling. If ``None``, a
            fresh unseeded generator is used.

    Returns:
        List[List[int]]: the windows. Each has length ``context_size`` except
        possibly the last, which may be shorter (the collator pads it).
    """
    if context_size <= 0:
        raise ValueError("context_size must be a positive integer")
    if not ids:
        return []
    pool = list(ids)
    (rng or random).shuffle(pool)
    return [pool[i : i + context_size] for i in range(0, len(pool), context_size)]


def tokenize_regionset(
    region_set,
    tokenizer,
    context_size: int = 8192,
    max_windows: Optional[int] = 10,
    rng: Optional[random.Random] = None,
    drop_unk: bool = True,
) -> List[List[int]]:
    """Tokenize one region set into a list of fixed-size token windows.

    Args:
        region_set: a ``gtars.models.RegionSet`` (or anything the tokenizer accepts).
        tokenizer: a ``gtars.tokenizers.Tokenizer`` (or ``TrainingTokenizer``); called
            as ``tokenizer(region_set)["input_ids"]``.
        context_size: tokens per window.
        max_windows: skip the file if it would produce more than this many windows
            (i.e. if the filtered token count exceeds ``max_windows * context_size``).
            ``None`` disables the cap.
        rng: optional ``random.Random`` for reproducible windowing.
        drop_unk: drop ``unk`` tokens (regions with no universe overlap) before
            windowing.

    Returns:
        List[List[int]]: token windows for this region set (empty if the file was
        skipped or produced no tokens).
    """
    ids = tokenizer(region_set)["input_ids"]

    if drop_unk:
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if unk_id is not None:
            ids = [i for i in ids if i != unk_id]

    if not ids:
        return []

    if max_windows is not None and len(ids) > max_windows * context_size:
        return []

    return sample_and_remove(ids, context_size, rng=rng)
