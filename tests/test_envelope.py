import numpy as np
import pytest

from fusion.envelope import reachable_symbols


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def test_three_line_example():
    pn = _softmax(np.array([0.0, 1.0, 2.0]))   # slopes in beta
    pl = _softmax(np.array([0.0, -0.5, -3.0]))  # intercepts
    # S0 = 0, S1 = beta - 0.5, S2 = 2 beta - 3: S1 beats S0 above 0.5, S2 beats S1 above 2.5
    assert reachable_symbols(pn, pl, 0.25, 4.0) == (0, 1, 2)
    assert reachable_symbols(pn, pl, 0.6, 2.0) == (1,)
    assert reachable_symbols(pn, pl, 0.25, 0.4) == (0,)
    assert reachable_symbols(pn, pl, 0.25, 2.4) == (0, 1)


def test_matches_dense_scan_on_random_posteriors():
    rng = np.random.default_rng(7)
    betas = np.geomspace(0.1, 20.0, 20001)
    mismatches = 0
    for _ in range(200):
        pn, pl = rng.dirichlet(np.full(36, 0.4)), rng.dirichlet(np.full(36, 0.4))
        scan = set(np.argmax(betas[:, None] * np.log(pn) + np.log(pl), axis=1))
        exact = set(reachable_symbols(pn, pl, 0.1, 20.0))
        assert scan <= exact
        mismatches += int(scan != exact)
    assert mismatches <= 2


def test_single_symbol_range_and_degenerate_slopes():
    pn = np.full(36, 1 / 36)  # equal slopes: neural evidence never changes the winner
    pl = _softmax(np.arange(36, dtype=float))
    assert reachable_symbols(pn, pl, 1e-6, 1e6) == (35,)


def _load_real(prior_model):
    """Working frame and attribution grid for one prior, or a skip when the posteriors are absent."""
    from fusion import data

    root = data._INTERMEDIATE
    needed = [root / "attribution.parquet", root / "selections.parquet",
              root / "prior_shards" / (prior_model.replace("/", "_") + ".parquet")]
    if not all(path.exists() for path in needed):
        pytest.skip("companion posteriors are not available")
    return data.build_working_frame(prior_model), data.load_attribution(prior_model)


@pytest.mark.parametrize("prior_model", ["gpt2", "ngram5_kn"])
def test_single_beta_matches_attribution_table_on_real_selections(prior_model):
    from fusion.data import SELECTION_KEY
    from fusion.rescue import BETA_GRID

    working, attribution = _load_real(prior_model)
    rng = np.random.default_rng(11)
    sample = working.iloc[rng.choice(len(working), size=500, replace=False)]
    key = list(SELECTION_KEY)
    table = attribution[key + ["beta", "fused_argmax"]]
    checked = 0
    for beta in BETA_GRID:
        at_beta = table[np.isclose(table["beta"], beta)].drop(columns="beta")
        joined = sample[key + ["p_neural", "p_lm"]].merge(at_beta, on=key, how="inner",
                                                          validate="one_to_one")
        assert len(joined) == len(sample)
        for row in joined.itertuples(index=False):
            got = reachable_symbols(row.p_neural, row.p_lm, beta, beta + 1e-9)
            assert got == (int(row.fused_argmax),)
            checked += 1
    assert checked == 500 * len(BETA_GRID)
