"""Tests for degree_of_recovery.core_batch (vectorised 3-D primitives)."""
from __future__ import annotations

import numpy as np
import pytest

from degree_of_recovery.core import knn_score, median_score
from degree_of_recovery.core_batch import (
    CAT_DEGRADED,
    CAT_INDISTINGUISHABLE,
    CAT_NAMES,
    CAT_NO_DATA,
    CAT_REGENERATING,
    bootstrap_ci_batch,
    classify_batch,
    cosine_dist_matrix,
    knn_mean_3d,
    median_3d,
)


class TestCosineDistMatrix:
    def test_self_distance_zero(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(5, 8))
        D = cosine_dist_matrix(A, A)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_shape(self):
        A = np.ones((3, 4))
        B = np.ones((7, 4))
        assert cosine_dist_matrix(A, B).shape == (3, 7)


class TestReductions3d:
    def test_median_3d_shape_and_values(self):
        D = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        out = median_3d(D)
        assert out.shape == (1, 2)
        np.testing.assert_allclose(out, [[2.0, 5.0]])

    def test_knn_mean_3d_picks_smallest(self):
        # Last axis: pick 2 smallest, mean them.
        D = np.array([[[5.0, 1.0, 3.0, 2.0]]])  # (1, 1, 4)
        out = knn_mean_3d(D, k=2)
        # smallest 2 are 1.0 and 2.0; mean = 1.5
        np.testing.assert_allclose(out, [[1.5]])


class TestBootstrapCiBatch:
    def test_point_matches_per_site_median(self):
        # Batched point should equal the per-site median_score for each probe.
        rng = np.random.default_rng(0)
        D_g = rng.uniform(0.1, 0.4, size=(4, 20))
        D_b = rng.uniform(0.6, 0.9, size=(4, 20))
        boot_rng = np.random.default_rng(42)
        point, lo, hi = bootstrap_ci_batch(D_g, D_b, "median", 100, boot_rng, k=5)

        for i in range(4):
            expected = median_score(D_g[i], D_b[i])
            assert point[i] == pytest.approx(expected, rel=1e-12)

    def test_point_matches_per_site_knn(self):
        rng = np.random.default_rng(0)
        D_g = rng.uniform(0.1, 0.4, size=(4, 20))
        D_b = rng.uniform(0.6, 0.9, size=(4, 20))
        boot_rng = np.random.default_rng(42)
        point, lo, hi = bootstrap_ci_batch(D_g, D_b, "knn", 100, boot_rng, k=5)

        for i in range(4):
            expected = knn_score(D_g[i], D_b[i], k=5)
            assert point[i] == pytest.approx(expected, rel=1e-12)

    def test_knn_returns_nan_when_too_few_refs(self):
        D_g = np.zeros((3, 4))
        D_b = np.zeros((3, 20))
        rng = np.random.default_rng(42)
        point, lo, hi = bootstrap_ci_batch(D_g, D_b, "knn", 50, rng, k=5)
        assert np.all(np.isnan(point))
        assert np.all(np.isnan(lo))
        assert np.all(np.isnan(hi))

    def test_ci_brackets_point(self):
        rng = np.random.default_rng(0)
        D_g = rng.uniform(0.1, 0.4, size=(5, 30))
        D_b = rng.uniform(0.6, 0.9, size=(5, 30))
        point, lo, hi = bootstrap_ci_batch(
            D_g, D_b, "median", 500, np.random.default_rng(42)
        )
        assert np.all(lo <= point) and np.all(point <= hi)

    def test_invalid_metric_raises(self):
        D_g = np.zeros((1, 10))
        D_b = np.zeros((1, 10))
        with pytest.raises(ValueError):
            bootstrap_ci_batch(
                D_g, D_b, "garbage", 10, np.random.default_rng(0)
            )

    def test_deterministic_with_same_seed(self):
        rng = np.random.default_rng(0)
        D_g = rng.uniform(0.1, 0.4, size=(3, 20))
        D_b = rng.uniform(0.6, 0.9, size=(3, 20))
        a = bootstrap_ci_batch(D_g, D_b, "knn", 200, np.random.default_rng(42))
        b = bootstrap_ci_batch(D_g, D_b, "knn", 200, np.random.default_rng(42))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


class TestClassifyBatch:
    def test_categories(self):
        # Order: regenerating, degraded, indistinguishable, no_data, indistinguishable (delta gate fails)
        score = np.array([0.70, 0.30, 0.50, np.nan, 0.55])
        lo = np.array([0.65, 0.20, 0.40, 0.40, 0.51])
        hi = np.array([0.75, 0.40, 0.60, 0.60, 0.59])
        out = classify_batch(score, lo, hi, t_lo=0.45, t_hi=0.55, delta=0.05)

        assert out[0] == CAT_REGENERATING     # lo > t_hi and score - t_hi >= delta
        assert out[1] == CAT_DEGRADED         # hi < t_lo and t_lo - score >= delta
        assert out[2] == CAT_INDISTINGUISHABLE
        assert out[3] == CAT_NO_DATA
        assert out[4] == CAT_INDISTINGUISHABLE  # lo > t_hi but score-t_hi < delta

    def test_decode_with_cat_names(self):
        score = np.array([0.7, 0.3])
        lo = np.array([0.65, 0.2])
        hi = np.array([0.75, 0.4])
        out = classify_batch(score, lo, hi, t_lo=0.45, t_hi=0.55, delta=0.05)
        names = CAT_NAMES[out]
        assert names[0] == "regenerating"
        assert names[1] == "degraded"
