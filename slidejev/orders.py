"""Map displayed candidate positions to their original zero-based indices."""

import hashlib
import random


ORDER_NAMES = ("original", "reversed", "cyclic", "random")


def option_orders(count, item_id, seed=0, *, random_order=None):
    """Return display-to-original permutations, optionally reusing a supplied shuffle."""
    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer of at least two")
    original = list(range(count))
    if random_order is None:
        # Use a per-question seed so batching and input-file order cannot change a run.
        digest = hashlib.sha256(f"{seed}\0{item_id}".encode("utf-8")).digest()
        shuffled = original.copy()
        random.Random(int.from_bytes(digest, "big")).shuffle(shuffled)
    else:
        shuffled = list(random_order)
        if any(type(i) is not int for i in shuffled) or sorted(shuffled) != original:
            raise ValueError("random_order must be a permutation of the candidate indices")
    return {
        "original": original,
        "reversed": original[::-1],
        "cyclic": original[1:] + original[:1],
        "random": shuffled,
    }
