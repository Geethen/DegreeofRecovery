import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic RNG matching the seed used by v3/v4 scorers."""
    return np.random.default_rng(42)


@pytest.fixture
def two_clouds_64d():
    """Two well-separated reference clouds in 64-D plus a probe near the good cloud.

    Mirrors the AlphaEarth setup (64-D embeddings, multi-row reference pools).
    The probe is intentionally placed near the good centroid so DoR > 0.5.
    """
    rng = np.random.default_rng(0)
    good = rng.normal(loc=1.0, scale=0.1, size=(20, 64))
    bad = rng.normal(loc=-1.0, scale=0.1, size=(20, 64))
    probe = good[0] + rng.normal(scale=0.05, size=64)
    return probe, good, bad
