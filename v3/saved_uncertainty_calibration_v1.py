"""Saved-output statistics only; no genotype, model or checkpoint readers.

Integer donor multiplicities represent frozen family bootstrap draws. Source
block multiplicities repeat whole alleles and calibration cells, never alter
within-allele truth. Missing values stay outside conditional statistics and in
denominators. Confidence is a saved unordered-call event probability; marginal
DS cannot substitute for it. Fixed reliability bins do not recalibrate models.
"""
import numpy as np

EDGES = np.arange(11, dtype=np.float64) / 10


def multiplicities(value, n):
    w = np.asarray(value, dtype=np.float64)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or (w != np.floor(w)).any():
        raise ValueError('complete nonnegative integer draw multiplicities required')
    if w.sum() > 2**52:
        raise ValueError('draw multiplicities exceed exact count representation')
    return w


def centered(value, weights):
    """Anchor before summation so an exactly constant vector is exactly zero."""
    x = np.asarray(value, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    delta = x - x[0]
    return delta - np.sum(w[:, None] * delta, axis=0) / w.sum()


def per_allele(dosages, truth, available, donor_draw):
    """Matched ALL-method cohort; shape methods x donors x source ALT alleles.

    At least three paired donor occurrences in a draw are required. Unique
    original donors are also returned, including draws with repeated families.
    No adaptive variance floor, smoothing or missing-as-zero convention.
    States: 0 defined, 1 fewer than three occurrences, 2 constant vector.
    """
    x = np.asarray(dosages, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    av = np.asarray(available)
    if x.ndim != 3 or y.shape != x.shape[1:] or av.shape != x.shape[:2] or av.dtype != bool:
        raise ValueError('explicit methods/donors/ALT and boolean availability required')
    m, n, a = x.shape
    if not m or not a:
        raise ValueError('nonempty complete method/ALT universe required')
    w = multiplicities(donor_draw, n)
    common = av.all(axis=0)
    if not np.isfinite(x[av]).all() or not np.isfinite(y[common]).all():
        raise ValueError('available paired dosage/truth must be finite')
    if ((x[av] < 0) | (x[av] > 2)).any() or not np.isin(y[common], [0, 1, 2]).all():
        raise ValueError('diploid per-ALT dosage range required')
    ix = np.flatnonzero(common & (w > 0))
    paired = int(w[ix].sum())
    result = dict(r2=np.full((m, a), np.nan), correlation=np.full((m, a), np.nan),
                  states=np.ones((m, a), dtype='u1'), paired_draw_occurrences=paired,
                  unique_paired_original_donors=len(ix), whole_draw_occurrences=int(w.sum()),
                  excluded_draw_occurrences=int(w[~common].sum()), common=common)
    if paired < 3:
        return result
    wy = w[ix]
    dy = centered(y[ix], wy)
    yy = np.sum(wy[:, None] * dy * dy, axis=0)
    for k in range(m):
        dx = centered(x[k, ix], wy)
        xx = np.sum(wy[:, None] * dx * dx, axis=0)
        xy = np.sum(wy[:, None] * dx * dy, axis=0)
        ok = (xx > 0) & (yy > 0)
        rho = (xy[ok] / np.sqrt(xx[ok])) / np.sqrt(yy[ok])
        if not np.isfinite(rho).all() or (np.abs(rho) > 1 + 1e-10).any():
            raise ValueError('paired correlation outside numerical range')
        result['states'][k] = np.where(ok, 0, 2)
        result['correlation'][k, ok] = rho
        result['r2'][k, ok] = rho * rho
    return result


def mean_defined_alleles(values, allele_draw):
    """Conditional macro-allele mean; block draws repeat full source alleles."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or np.isinf(v).any():
        raise ValueError('one finite-or-undefined statistic per source allele required')
    w = multiplicities(allele_draw, len(v))
    ok = np.isfinite(v)
    if ((v[ok] < 0) | (v[ok] > 1 + 1e-10)).any():
        raise ValueError('r2 range required')
    den = int(w[ok].sum())
    return dict(state='DEFINED_CONDITIONAL' if den else 'UNAVAILABLE_NO_DEFINED_ALLELE',
                value=float(np.sum(w[ok] * v[ok]) / den) if den else None,
                whole_allele_draw_occurrences=int(w.sum()), defined_allele_draw_occurrences=den,
                undefined_allele_draw_occurrences=int(w[~ok].sum()))


def paired_allele_contrast(left, right, allele_draw):
    """Difference on the same defined alleles, never subtract separate means."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError('same ordered finite-or-undefined allele universe required')
    w = multiplicities(allele_draw, len(a))
    ok = np.isfinite(a) & np.isfinite(b)
    for v in [a, b]:
        finite = v[np.isfinite(v)]
        if ((finite < 0) | (finite > 1 + 1e-10)).any():
            raise ValueError('r2 range required')
    den = int(w[ok].sum())
    return dict(state='DEFINED_PAIRED_ALLELE_CONDITIONAL' if den else 'UNAVAILABLE_NO_PAIRED_DEFINED_ALLELE',
                difference=float(np.sum(w[ok]*(a[ok]-b[ok]))/den) if den else None,
                whole_allele_draw_occurrences=int(w.sum()), paired_defined_allele_draw_occurrences=den,
                unpaired_or_undefined_allele_draw_occurrences=int(w[~ok].sum()))


def reliability(confidence, correct, available, cell_draw):
    """One method/cohort, exact saved call event; fixed [lo,hi), final hi=1.

    Return bin count/confidence/correct sums so family/source-block sufficient
    statistics can be combined without rereading predictions. Brier here is
    binary *call-correctness* loss, not the full genotype-distribution Brier.
    """
    p = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    av = np.asarray(available)
    if p.ndim != 1 or y.shape != p.shape or av.shape != p.shape or av.dtype != bool:
        raise ValueError('complete saved confidence/call/availability vector required')
    w = multiplicities(cell_draw, len(p))
    if not np.isfinite(p[av]).all() or not np.isfinite(y[av]).all():
        raise ValueError('available confidence and call correctness must be finite')
    if ((p[av] < 0) | (p[av] > 1)).any() or not np.isin(y[av], [0, 1]).all():
        raise ValueError('exact event probability and binary correctness required')
    ix = np.flatnonzero(av & (w > 0))
    bins = np.minimum(np.searchsorted(EDGES, p[ix], side='right') - 1, 9)
    sums = np.zeros((10, 3), dtype=np.float64)
    for col, v in enumerate([np.ones(len(ix)), p[ix], y[ix]]):
        np.add.at(sums[:, col], bins, w[ix] * v)
    loss = float(np.sum(w[ix] * (p[ix] - y[ix])**2))
    return dict(bin_sums=sums, call_correctness_Brier_sum=loss,
                whole_draw_occurrences=int(w.sum()), available_draw_occurrences=int(w[ix].sum()),
                unavailable_draw_occurrences=int(w[~av].sum()))


def reliability_summary(bin_sums, call_correctness_Brier_sum):
    s = np.asarray(bin_sums, dtype=np.float64)
    loss = float(call_correctness_Brier_sum)
    if s.shape != (10, 3) or not np.isfinite(s).all() or (s < 0).any() or not np.isfinite(loss) or loss < 0:
        raise ValueError('finite nonnegative frozen reliability sufficient statistics required')
    n = s[:, 0]
    if (n != np.floor(n)).any() or (s[:, 1:] > n[:, None] + 1e-10).any() or loss > n.sum() + 1e-10:
        raise ValueError('reliability counts/probabilities/correctness bounds required')
    den = int(n.sum())
    bins = [dict(lower=float(EDGES[i]), upper=float(EDGES[i+1]), upper_inclusive=i == 9,
                 count=int(n[i]), mean_confidence=float(s[i, 1]/n[i]) if n[i] else None,
                 observed_correct_fraction=float(s[i, 2]/n[i]) if n[i] else None) for i in range(10)]
    return dict(state='DEFINED_AVAILABLE_CONDITIONAL' if den else 'UNAVAILABLE_NO_CALL',
                available_draw_occurrences=den, bins=bins,
                fixed_bin_ECE=float(np.abs(s[:, 1]-s[:, 2]).sum()/den) if den else None,
                call_correctness_Brier=float(loss/den) if den else None)


def conditional_percentiles(values):
    """Nearest-rank descriptive percentiles; undefined draws never removed silently."""
    v = list(values)
    finite = sorted(float(x) for x in v if x is not None)
    if not all(np.isfinite(x) for x in finite):
        raise ValueError('undefined draw is None; infinite r2/calibration is invalid')
    q = [finite[max(0, int(np.ceil(f*len(finite)))-1)] if finite else None for f in [.025, .5, .975]]
    return dict(total_draws=len(v), defined_draws=len(finite), undefined_draws=len(v)-len(finite),
                nearest_rank_percentiles=q,
                interpretation='Conditional on defined draws; descriptive dependence sensitivity, not unconditional or joint simultaneous confidence intervals. Source 1Mb blocks are not validated LD blocks.')
