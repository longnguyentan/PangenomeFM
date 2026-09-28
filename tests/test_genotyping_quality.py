import pandas as pd
import pytest

from tasks.transfer.genotyping_quality import prepare_quality


def inputs():
    regions = pd.DataFrame([['chr1', 0, 10, 'A'], ['chr2', 20, 30, 'B']],
                           columns=['chrom', 'start', 'end', 'gene_name'])
    row = dict(sample='sample1', region='chr1_0_10', gene_name='A', QV_1_pred=20.,
               QV_2_pred=30., QV_sum_pred=50., QV_1_best=30., QV_2_best=40.,
               QV_sum_best=70., avg_error_rate_pred=.0055)
    return pd.DataFrame([row, row]), regions


def test_exact_duplicates_do_not_reweight_loci_and_missing_is_unmeasured():
    f, b = inputs()
    rows, loci, missing, qc = prepare_quality(f, b)
    assert len(rows) == 1 and qc['exact_duplicate_rows_removed'] == 1
    assert loci.n_observed_donors.tolist() == [1, 0]
    assert pd.isna(loci.iloc[1].mean_qv_fraction)
    assert missing.locus_id.tolist() == ['chr2_20_30']
    assert rows.qv_fraction.iloc[0] == pytest.approx(5/7)


@pytest.mark.parametrize('field,value', [('QV_sum_pred', 51), ('gene_name', 'wrong'),
                                        ('region', 'chr1_1_10'), ('avg_error_rate_pred', -1)])
def test_bad_quality_or_identity_rejected(field, value):
    f, b = inputs()
    f = f.iloc[:1].copy()
    f.loc[0, field] = value
    with pytest.raises(ValueError):
        prepare_quality(f, b)


def test_conflicts_rejected_instead_of_selecting_better_measurement():
    f, b = inputs()
    f.loc[1, 'avg_error_rate_pred'] = .01
    with pytest.raises(ValueError, match='Conflicting'):
        prepare_quality(f, b)


def test_source_rounding_retained_but_large_best_match_violation_rejected():
    f, b = inputs()
    f = f.iloc[:1].copy()
    f.loc[0, ['QV_1_best', 'QV_2_best', 'QV_sum_best']] = [20, 29.9999, 49.9999]
    rows, _, _, qc = prepare_quality(f, b)
    assert rows.qv_fraction.iloc[0] > 1 and qc['qv_fraction_above_one'] == 1
    f.loc[0, ['QV_2_best', 'QV_sum_best']] = [29.9, 49.9]
    with pytest.raises(ValueError, match='rounding'):
        prepare_quality(f, b)
