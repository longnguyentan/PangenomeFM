import pandas as pd
import pytest

from tasks.transfer.sv_types import CLASSES, matched_subset, normalize


def fixture():
    return pd.DataFrame([{'ID': f'chr1-{101+i}-{kind}-{length}', '#CHROM': 'chr1', 'POS': 100+i,
        'END': 101+i if kind == 'INS' else 100+i+length, 'SVTYPE': kind, 'SVLEN': length}
        for kind in CLASSES for i, length in enumerate([100, 120, 140, 900])])


def test_classes_and_first_affected_base_are_preserved():
    data = normalize(fixture())
    assert data.label.nunique() == 3 and len(data) == 12
    assert data.end.eq(data.start+1).all()
    assert data.loc[data.svtype.eq('INV'), 'label'].eq(2).all()
    assert data.groupby('locus_id').label.nunique().eq(3).all()
    assert 'binary_svtype_label' not in data
    for field, value in [('POS', 99), ('SVTYPE', 'DUP'), ('SVLEN', 10), ('END', 100.2)]:
        bad = fixture()
        if field == 'END':
            bad[field] = bad[field].astype(float)
        bad.loc[0, field] = value
        with pytest.raises(ValueError):
            normalize(bad)
    with pytest.raises(ValueError, match='Duplicate'):
        normalize(pd.concat([fixture(), fixture().iloc[[0]]]))


def test_matching_is_stable_complete_and_without_replacement():
    data = normalize(fixture())
    data = data.drop(data.index[(data.svtype == 'INV') & (data.svlen == 120)])
    matched, strata = matched_subset(data, 42)
    shuffled, _ = matched_subset(data.sample(frac=1, random_state=13), 42)
    pd.testing.assert_frame_equal(matched, shuffled)
    assert not matched.variant_id.duplicated().any()
    assert set(matched.variant_id) <= set(data.variant_id)
    for _, group in matched.groupby(['chrom', 'length_bin']):
        assert group.svtype.value_counts().nunique() == 1
        assert set(group.svtype) == set(CLASSES)
    assert len(matched) == 3*strata.n_per_class.sum()
    with pytest.raises(ValueError, match='common support'):
        matched_subset(data.loc[data.svtype.ne('INV')], 42)


def test_versioned_and_retained_ids_do_not_override_bed_positions():
    source = fixture()
    source.loc[0, 'ID'] = 'chr1-101-DEL-100.1'
    source.loc[4, 'ID'] = 'chr1-100-INS-100'
    result = normalize(source)
    assert result.loc[result.variant_id.eq('chr1-100-INS-100'), 'start'].item() == 100
    assert result.loc[result.variant_id.eq('chr1-100-INS-100'), 'id_position_minus_bed_start'].item() == 0


def test_padded_vcf_independently_confirms_bed_anchor(tmp_path):
    import gzip
    from tasks.transfer.sv_types import validate_padded_vcf
    examples = normalize(fixture())
    records = ['##fileformat=VCFv4.2']
    for row in examples.loc[examples.svtype.ne('INV')].itertuples():
        ref, alt = ('A', 'A'+'C'*row.svlen) if row.svtype == 'INS' else ('A'+'C'*row.svlen, 'A')
        records.append(f'{row.chrom}\t{row.start}\t{row.variant_id}\t{ref}\t{alt}\t.\tPASS\t.')
    path = tmp_path/'variants.vcf.gz'
    with gzip.open(path,'wt') as handle:
        handle.write('\n'.join(records)+'\n')
    assert validate_padded_vcf(examples, path)['n_events'] == 8
    with pytest.raises(ValueError, match='coordinate mismatch'):
        validate_padded_vcf(examples.assign(start=examples.start-1), path)
