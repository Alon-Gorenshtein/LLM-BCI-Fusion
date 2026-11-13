"""Exact set of symbols that can be the fused argmax for some beta in a range. The fused log-score
S_j(beta) = beta * a_j + b_j is linear in beta (a = log neural, b = log language model), so the
winner over beta is the upper envelope of 36 lines; walk it from the left end."""
from __future__ import annotations

import numpy as np

from fusion.data import AUTHORSHIP_ROOT  # noqa: F401  (puts the companion package on sys.path)

from authorship.attribution import PROBABILITY_FLOOR  # noqa: E402  (the same floor fuse() applies)

_TOL = 1e-12


def reachable_symbols(p_neural, p_lm, beta_lo: float, beta_hi: float) -> tuple[int, ...]:
    a = np.log(np.maximum(np.asarray(p_neural, float), PROBABILITY_FLOOR))
    b = np.log(np.maximum(np.asarray(p_lm, float), PROBABILITY_FLOOR))
    beta = float(beta_lo)
    scores = beta * a + b
    tied = np.flatnonzero(scores >= scores.max() - _TOL)
    w = int(tied[np.argmax(a[tied])])            # on a tie the steeper line wins as beta grows
    seen = {w}
    while True:
        steeper = np.flatnonzero(a > a[w] + _TOL)
        if steeper.size == 0:
            break
        cross = (b[w] - b[steeper]) / (a[steeper] - a[w])
        cross = np.where(cross > beta + _TOL, cross, np.inf)
        nxt = float(cross.min())
        if not np.isfinite(nxt) or nxt >= beta_hi - _TOL:
            break
        near = steeper[cross <= nxt + _TOL]
        w = int(near[np.argmax(a[near])])
        beta = nxt
        seen.add(w)
    return tuple(sorted(seen))
