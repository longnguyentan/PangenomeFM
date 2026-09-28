import numpy as np
import pandas as pd
import pytest

from scripts.server.audit_reference_component_context import component_context


def test_alternative_components_keep_full_denominator_and_ambiguous_anchors():
    # Ref A/B are chr1, C is chr2. X-Y is a partially cached two-anchor component.
    nodes = pd.DataFrame(dict(name=list('ABCXYZWU'),
        SN=['GRCh38#0#chr1']*2+['GRCh38#0#chr2']+['sample#1#chr1']*5,
        SO=[0,100,0,0,0,0,0,0], LN=[10]*8))
    links = pd.DataFrame([('A','X'),('X','Y'),('Y','B'),('B','Y'),('A','Z'),('Z','C'),('B','W')],
                         columns=['from_seg','to_seg'])
    frame = component_context(nodes,links,np.array([0,1,2,3,5,6]))
    assert frame.n_alternative_segments.sum()==5
    assert len(frame)==4
    paired = frame.loc[frame.status.eq('multiple_anchors_one_primary_chromosome')].iloc[0]
    assert paired.n_alternative_segments==2 and paired.n_cached==1 and paired.coverage=='partial'
    assert paired.n_reference_anchors==2 and paired.anchor_span_bp==110
    assert frame.status.value_counts().to_dict()==dict(multiple_anchors_one_primary_chromosome=1,
        multiple_reference_contigs=1,single_reference_anchor=1,unanchored=1)
    assert frame.loc[frame.status.eq('multiple_reference_contigs'),'anchor_span_bp'].isna().all()
    assert frame.loc[frame.status.eq('unanchored'),'coverage'].iloc[0]=='absent'
    with pytest.raises(ValueError,match='canonical rows'):
        component_context(nodes,links,np.array([0,0]))


def test_no_boundary_alternatives_are_unanchored_and_nonprimary_is_explicit():
    nodes = pd.DataFrame(dict(name=['r','a'],SN=['GRCh38#0#chrUn','sample#1#chr1'],SO=[0,0],LN=[1,2]))
    empty = pd.DataFrame(columns=['from_seg','to_seg'])
    result = component_context(nodes,empty,np.array([0]))
    assert result.iloc[0].status=='unanchored'
    links = pd.DataFrame(dict(from_seg=['r'],to_seg=['a']))
    result = component_context(nodes,links,np.array([0]))
    assert result.iloc[0].status=='nonprimary_reference_contig'
