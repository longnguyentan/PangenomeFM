import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tasks.transfer.controls import chromosome_permutation
from tasks.transfer.eqtl import gene_tss, prepare_tissue
from tasks.transfer.hg008 import in_regions, normalize, verify_md5
from tasks.transfer.external_sv import fit_original, masks


def test_completed_entex_artifact_regression():
    manifest = json.loads(
        Path("configs/completed_entex_regression_20260922.json").read_text()
    )
    assert len(manifest["sha256"]) > 20
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name


def test_hg008_clonal_and_original_endpoint_convention(tmp_path):
    rows = [
        "chr1\t11\ti\tA\t<INS>\t.\tPASS\tSVTYPE=INS;SVLEN=50;SUBCLONAL=n",
        "chr1\t20\td\tA\t<DEL>\t.\tPASS\tSVTYPE=DEL;SVLEN=-50;END=70;SUBCLONAL=n",
        "chr1\t30\ts\tA\t<INS>\t.\tPASS\tSVTYPE=INS;SVLEN=50;SUBCLONAL=y",
        "chr1\t30\tb\tA\t<BND>\t.\tPASS\tSVTYPE=BND;SUBCLONAL=n",
        "chr1\t100\to\tA\t<INS>\t.\tPASS\tSVTYPE=INS;SVLEN=50;SUBCLONAL=n",
    ]
    path = tmp_path / "input.vcf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(rows) + "\n")
    bed = pd.DataFrame({"chrom": ["chr1"], "start": [10], "end": [99]})
    x, excluded, qc = normalize(path, bed)
    assert x.variant_id.tolist() == ["i", "d"]
    assert x.end.tolist() == [60, 70]  # manuscript inferred insertion pseudo-end
    assert x.biological_end0.tolist() == [10, 69]
    assert qc["positive_prevalence"] == 0.5
    assert set(excluded.reason) == {
        "subclonal_or_unspecified",
        "not_INS_or_DEL",
        "biological_endpoint_outside_clonal_BED",
    }
    assert in_regions(bed, "chr1", 10) and not in_regions(bed, "chr1", 99)


def test_release_checksum(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"source")
    checksum = tmp_path / "md5"
    checksum.write_text("source " + hashlib.md5(b"source").hexdigest())
    verify_md5(path, checksum)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_md5(path, checksum)


def eqtl_fixture():
    cfg = json.loads(Path("configs/downstream_transfer_v1.json").read_text())["eqtl"]
    x = pd.DataFrame(
        {
            "phenotype_id": ["g"] * 4,
            "variant_id": [f"chr1_{x}_A_C_b38" for x in [10, 20, 30, 40]],
            "pip": [0.99, 0.001, 0.5, 0.002],
            "af": [0.15] * 4,
            "cs_id": [1, 2, 2, 2],
        }
    )
    genes = pd.DataFrame({"phenotype_id": ["g"], "gene_chrom": ["chr1"], "tss0": [0]})
    return x, genes, cfg


def test_eqtl_matching_deterministic_unique_and_gene_grouped():
    raw, genes, cfg = eqtl_fixture()
    x, pairs, qc = prepare_tissue(raw, genes, cfg, "Whole_Blood")
    y, _, _ = prepare_tissue(
        raw.sample(frac=1, random_state=42), genes, cfg, "Whole_Blood"
    )
    assert x.locus_id.tolist() == y.locus_id.tolist()
    assert len(x) == 2 and not x.locus_id.duplicated().any()
    assert x.start.tolist() == [int(v.split(":")[1]) - 1 for v in x.locus_id]
    assert x.groupby("match_pair").label.sum().eq(1).all()
    assert x.groupby("match_pair").phenotype_id.nunique().eq(1).all()
    assert qc["excluded_intermediate_pip_rows"] == 1


def test_eqtl_conflicting_locus_cannot_be_control():
    raw, genes, cfg = eqtl_fixture()
    conflict = raw.iloc[[0]].assign(pip=0.001, phenotype_id="other")
    with pytest.raises(ValueError, match="No high/low"):
        prepare_tissue(pd.concat([raw, conflict]), genes, cfg, "Whole_Blood")


def test_gene_tss_strand_and_version(tmp_path):
    p = tmp_path / "genes.gtf"
    p.write_text(
        'chr1\tx\tgene\t10\t20\t.\t+\t.\tgene_id "ENSG.1";\nchr2\tx\tgene\t30\t40\t.\t-\t.\tgene_id "ENSG.2";\n'
    )
    assert gene_tss(p).tss0.tolist() == [9, 39]


def test_controls_do_not_cross_chromosomes_or_splits():
    chromosomes = np.repeat(["chr1", "chr2", "chr3"], 20)
    order = chromosome_permutation(chromosomes, 42)
    assert np.array_equal(chromosomes[order], chromosomes)
    assert sorted(order) == list(range(60))
    train, val, test = masks(chromosomes, {"chr3"}, {"chr2"})
    assert (train.sum(), val.sum(), test.sum()) == (20, 20, 20)
    assert np.array_equal(train[order], train)
    with pytest.raises(ValueError, match="Overlapping"):
        masks(chromosomes, {"chr3"}, {"chr3"})


def test_external_fit_sees_only_original_train_validation():
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(90, 3))
    y = np.tile([0, 1], 45)
    train, val, test = masks(
        np.repeat(["chr1", "chr2", "chr3"], 30), {"chr3"}, {"chr2"}
    )
    model, temp, threshold = fit_original(matrix, y, train, val, 42)
    changed = y.copy()
    changed[test] = 1 - changed[test]
    other, temp2, threshold2 = fit_original(matrix, changed, train, val, 42)
    np.testing.assert_array_equal(
        model.predict_proba(matrix), other.predict_proba(matrix)
    )
    assert temp == temp2 and threshold == threshold2
