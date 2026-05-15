"""Known-answer tests for degree_of_recovery.core (per-site primitives)."""
from __future__ import annotations

import numpy as np
import pytest

from degree_of_recovery.core import (
    EPS,
    bootstrap_ci,
    classify,
    cosine_dist,
    cosine_dists_to_set,
    knn_score,
    median_score,
)


class TestCosine:
    def test_identical_vectors_zero_distance(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_dist(v, v) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors_unit_distance(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_dist(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_antipodal_vectors_two_distance(self):
        a = np.array([1.0, 1.0, 1.0])
        assert cosine_dist(a, -a) == pytest.approx(2.0, abs=1e-10)

    def test_dists_to_set_matches_per_row_loop(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=10)
        pts = rng.normal(size=(7, 10))
        bulk = cosine_dists_to_set(x, pts)
        loop = np.array([cosine_dist(x, p) for p in pts])
        np.testing.assert_allclose(bulk, loop, atol=1e-12)


class TestKnnScore:
    def test_zero_when_probe_at_good_centroid(self):
        # All d_g near 0, all d_b near 1 -> score -> 1.0 (probe deep in good).
        d_g = np.full(10, 0.01)
        d_b = np.full(10, 0.99)
        score = knn_score(d_g, d_b, k=5)
        assert score == pytest.approx(0.99 / (0.01 + 0.99 + EPS), rel=1e-9)

    def test_one_when_probe_at_bad_centroid(self):
        d_g = np.full(10, 0.99)
        d_b = np.full(10, 0.01)
        score = knn_score(d_g, d_b, k=5)
        assert score < 0.05

    def test_half_when_equidistant(self):
        d_g = np.full(10, 0.5)
        d_b = np.full(10, 0.5)
        score = knn_score(d_g, d_b, k=5)
        assert score == pytest.approx(0.5, abs=1e-9)

    def test_uses_only_k_smallest(self):
        # k=3, smallest 3 of d_g are [0.1, 0.1, 0.1] -> mean 0.1
        # smallest 3 of d_b are [0.5, 0.5, 0.5] -> mean 0.5
        d_g = np.array([0.1, 0.1, 0.1, 9.0, 9.0])
        d_b = np.array([0.5, 0.5, 0.5, 9.0, 9.0])
        score = knn_score(d_g, d_b, k=3)
        assert score == pytest.approx(0.5 / (0.1 + 0.5 + EPS), rel=1e-9)

    def test_nan_when_too_few_refs(self):
        assert np.isnan(knn_score(np.array([0.1, 0.2]), np.full(10, 0.5), k=5))
        assert np.isnan(knn_score(np.full(10, 0.5), np.array([0.1, 0.2]), k=5))


class TestMedianScore:
    def test_half_when_equal_medians(self):
        assert median_score(np.full(5, 0.3), np.full(5, 0.3)) == pytest.approx(
            0.5, abs=1e-9
        )

    def test_robust_to_outliers(self):
        # Median is robust; one big outlier in d_g shouldn't move score much.
        d_g = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 100.0])
        d_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        # median(d_g) = 0.1, median(d_b) = 0.5
        assert median_score(d_g, d_b) == pytest.approx(
            0.5 / (0.1 + 0.5 + EPS), rel=1e-9
        )


class TestBootstrapCi:
    def test_deterministic_with_same_seed(self, two_clouds_64d):
        probe, good, bad = two_clouds_64d
        d_g = cosine_dists_to_set(probe, good)
        d_b = cosine_dists_to_set(probe, bad)

        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        a = bootstrap_ci(median_score, d_g, d_b, n_boot=200, rng=rng_a)
        b = bootstrap_ci(median_score, d_g, d_b, n_boot=200, rng=rng_b)
        assert a == b

    def test_point_matches_score_fn(self, two_clouds_64d):
        probe, good, bad = two_clouds_64d
        d_g = cosine_dists_to_set(probe, good)
        d_b = cosine_dists_to_set(probe, bad)
        rng = np.random.default_rng(42)
        point, lo, hi = bootstrap_ci(median_score, d_g, d_b, 200, rng)
        assert point == pytest.approx(median_score(d_g, d_b), rel=1e-12)

    def test_ci_brackets_point(self, two_clouds_64d):
        probe, good, bad = two_clouds_64d
        d_g = cosine_dists_to_set(probe, good)
        d_b = cosine_dists_to_set(probe, bad)
        rng = np.random.default_rng(42)
        point, lo, hi = bootstrap_ci(median_score, d_g, d_b, 500, rng)
        assert lo <= point <= hi

    def test_nan_when_score_fn_returns_nan(self):
        # knn_score with too few refs returns NaN; bootstrap_ci must propagate.
        rng = np.random.default_rng(42)
        d_g = np.array([0.1, 0.2])
        d_b = np.full(10, 0.5)
        point, lo, hi = bootstrap_ci(
            lambda g, b: knn_score(g, b, k=5), d_g, d_b, 50, rng
        )
        assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


class TestClassify:
    def test_recovering_when_ci_fully_above_threshold(self):
        assert classify(0.7, 0.6, 0.8, threshold=0.5) == "recovering"

    def test_degraded_when_ci_fully_below_threshold(self):
        assert classify(0.3, 0.2, 0.4, threshold=0.5) == "degraded"

    def test_indistinguishable_when_ci_straddles_threshold(self):
        assert classify(0.5, 0.4, 0.6, threshold=0.5) == "indistinguishable"

    def test_no_data_on_nan_score(self):
        assert classify(np.nan, 0.4, 0.6, threshold=0.5) == "no_data"

    def test_no_data_on_nan_ci(self):
        assert classify(0.5, np.nan, 0.6, threshold=0.5) == "no_data"
        assert classify(0.5, 0.4, np.nan, threshold=0.5) == "no_data"

    def test_boundary_lo_equals_threshold_is_indistinguishable(self):
        # ci_lo > threshold (strict), so equality stays indistinguishable.
        assert classify(0.6, 0.5, 0.7, threshold=0.5) == "indistinguishable"
