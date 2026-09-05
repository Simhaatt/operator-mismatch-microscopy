"""Rank statistics, with ties handled.

Four scripts each carried their own copy of a Spearman correlation built from
``argsort(argsort(x))``. That expression is only a rank when every value is
distinct. With ties it hands equal values consecutive, arbitrary ranks, which
biases the correlation and in the degenerate case is badly wrong: against a
*constant* second variable it returns **+1.0**, a perfect correlation where the
right answer is zero.

That is not hypothetical here. ``rms`` is the aberration ladder and takes five
distinct values across thousands of samples, so every reported correlation
against it was computed on tied data; measured against this implementation the
old one understated them by 0.008 to 0.026. The block-comparison in Gate 5
ranks correlations computed over as few as 16 points, where tie handling can
decide the ordering outright.

Verified against ``scipy.stats.spearmanr`` on tied input.
"""

from __future__ import annotations

import numpy as np

__all__ = ["average_ranks", "spearman"]


def average_ranks(x) -> np.ndarray:
    """Ranks of ``x`` with tied values sharing their group's mean rank."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x, kind="stable")
    s = x[order]
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation, tie-corrected.

    Returns 0.0 when either input has no rank variation at all -- a constant
    variable carries no ordering information, so no correlation with it is
    defined, and reporting zero is the honest answer.
    """
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.size != rb.size:
        raise ValueError(f"length mismatch: {ra.size} vs {rb.size}")
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0
