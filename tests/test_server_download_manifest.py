from pathlib import Path

from scripts.server.download_full_data import load_manifest, select_profile


ROOT = Path(__file__).resolve().parents[1]


def test_server_manifest_profiles_are_nested_and_unique():
    resources = load_manifest(ROOT / "configs/server_full_data_manifest.tsv")
    ids = [resource.resource_id for resource in resources]
    assert len(ids) == len(set(ids))
    core = {resource.resource_id for resource in select_profile(resources, "sv-core")}
    analysis = {
        resource.resource_id for resource in select_profile(resources, "analysis-full")
    }
    archive = {
        resource.resource_id
        for resource in select_profile(resources, "archive-full-resolution")
    }
    assert core < analysis < archive
    assert {
        "hprc_r2_sv_gfa",
        "hgsvc3_sv_gfa",
        "hprc_r1_1_sv_gfa",
        "hgsvc3_hprc1_combined_sv_gfa",
    } <= core
    downstream = {
        resource.resource_id for resource in select_profile(resources, "downstream-sv")
    }
    assert {
        "hprc_r2_wave_vcf",
        "hgsvc3_sv_alt_vcf",
        "hgsvc3_sv_annotation",
        "hgsvc3_inv_vcf",
    } <= downstream


def test_every_resource_has_safe_relative_destination_and_https_url():
    resources = load_manifest(ROOT / "configs/server_full_data_manifest.tsv")
    for resource in resources:
        destination = Path(resource.relative_path)
        assert not destination.is_absolute()
        assert ".." not in destination.parts
        assert resource.url.startswith("https://")
        assert resource.expected_bytes is None or resource.expected_bytes > 0
