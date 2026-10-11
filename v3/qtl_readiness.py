"""Strict readiness, identity and effect orientation checks; no QTL claim."""
from collections import Counter
import math

MAGE_UNRESOLVED_2026_LIBRARIES={'SRR19762530','SRR19762653'}

def validate_sample_join(expression_samples, dosage_samples, covariate_samples, metadata, *, expected_samples, mage_identity_correction):
    groups=(expression_samples,dosage_samples,covariate_samples)
    if any(len(x)!=len(set(x)) for x in groups):raise ValueError('duplicate biological sample; technical libraries are not independent donors')
    if any(list(x)!=list(expected_samples) for x in groups):raise ValueError('exact full cohort order required; no silent sample intersection/pruning')
    if len(metadata)!=len(expected_samples) or {r['sample'] for r in metadata}!=set(expected_samples):raise ValueError('exact biological metadata join required')
    if any(r['library'] in MAGE_UNRESOLVED_2026_LIBRARIES for r in metadata) or mage_identity_correction.get('all_libraries_verified') is not True:
        raise ValueError('MAGE corrected sample/library identity must be resolved before QTL mapping')
    if any(r.get('tissue')!='LCL' for r in metadata):raise ValueError('declared MAGE LCL measurement required')
    return dict(samples=len(expected_samples),biological_unit='DONOR',ready_for_identity_join_only=True)

def orient_effect(ref,alt,effect_allele,other_allele,beta,frequency):
    if not all(isinstance(v,str) and v for v in (ref,alt,effect_allele,other_allele)) or ref==alt:
        raise ValueError('exact full literal REF/ALT identities required')
    if not math.isfinite(beta) or not math.isfinite(frequency) or not 0<=frequency<=1:raise ValueError('finite source effect and frequency required')
    if (effect_allele,other_allele)==(alt,ref):return beta,frequency
    if (effect_allele,other_allele)==(ref,alt):return -beta,1-frequency
    raise ValueError('alleles/build/strand unresolved; no guessed normalization or ambiguous SNP rescue')

def stage_gate(evidence,stage):
    common=('full_sample_identity','reference_and_alleles','complete_covariates','IBD_and_relatedness','missingness_denominators')
    required=common+({'QTL':('calibrated_cis_mapping','matched_sample_variant_comparison','replication'),
                     'SMR':('GWAS_summary_identity','eQTL_summary_identity','matched_LD','allele_orientation','HEIDI')}.get(stage,()))
    if stage not in ('QTL','SMR'):raise ValueError('declared endpoint required')
    pending=[k for k in required if evidence.get(k) is not True]
    return dict(stage=stage,ready=not pending,pending=pending,causality_demonstrated=False)
