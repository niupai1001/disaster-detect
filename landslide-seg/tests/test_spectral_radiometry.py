import pytest
import torch
from landslide_seg.models.spectral_contract import harmonized_reflectance, spectral_indices
from landslide_seg.models.spectral_ladder import SpectralLadder
from landslide_seg.models.spectral_fusion import SpectralBranch
from landslide_seg.models.spectral_temporal import BandRelationBlock


def test_three_paths_share_explicit_physical_definition():
    stats = {"mean": [2000.] * 10, "std": [500.] * 10}
    dn = torch.linspace(0, 10000, 160).reshape(1, 10, 4, 4)
    x = (dn - 2000) / 500
    mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
    rho = harmonized_reflectance(x, mean, std)
    assert torch.allclose(rho, dn / 10000, atol=1e-7)
    ladder = SpectralLadder("S1", "treatment", stats, radiometry="harmonized_dn_v1")
    branch = SpectralBranch("indices", "gated", 16, radiometry="harmonized_dn_v1")
    relation = BandRelationBlock(stats=stats, width=16, heads=4, layers=1)
    assert torch.equal(relation.reflectance(x), rho)
    assert torch.equal(ladder._indices(x), spectral_indices(rho))
    assert torch.equal(branch.spectral_maps(x, mean[None,:,None,None], std[None,:,None,None]), spectral_indices(rho))


def test_legacy_indices_remain_dn_based_and_new_version_is_explicit():
    stats = {"mean": [1.] * 10, "std": [1.] * 10}
    x = torch.zeros(1, 10, 4, 4)
    x[:, 7] = 2
    old = SpectralLadder("S1", "treatment", stats)
    new = SpectralLadder("S1", "treatment", stats, radiometry="harmonized_dn_v1")
    assert old._indices(x)[0,0,0,0] == pytest.approx(2 / 4.0001)
    assert new._indices(x)[0,0,0,0] == pytest.approx(.4)
    assert old.report()["radiometry"] == "legacy_dn"


def test_no_second_offset_and_zero_finite():
    dn = torch.zeros(1, 10, 4, 4)
    rho = harmonized_reflectance(dn, torch.zeros(10), torch.ones(10))
    assert torch.equal(rho, dn)
    assert torch.isfinite(spectral_indices(rho)).all()
    assert torch.count_nonzero(spectral_indices(rho)) == 0


def test_unknown_radiometry_rejected():
    with pytest.raises(ValueError):
        SpectralBranch("indices", "gated", 16, radiometry="auto")
