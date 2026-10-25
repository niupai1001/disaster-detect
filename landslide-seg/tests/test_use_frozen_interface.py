import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_use_frozen_interface import check_interface
from landslide_seg.data.optical_observation import OpticalObservation
from landslide_seg.models.frozen_use import FrozenUSE
from landslide_seg.models.universal_spectral_encoder import UniversalSpectralEncoder


def test_full_interface_checks_with_variable_bands():
    torch.manual_seed(1234)
    x = torch.randn(6, 9, 11)
    observation = OpticalObservation(x, tuple('abcdef'), torch.ones(6, 3),
        torch.ones(6, 3, dtype=torch.bool), torch.ones(6, dtype=torch.bool),
        torch.ones_like(x, dtype=torch.bool), {})
    frozen = FrozenUSE(UniversalSpectralEncoder())
    proof = check_interface(frozen, [('three', observation.subset('abc')), ('six', observation)], 'cpu')
    assert proof['passed'] and proof['mixed_sensor_and_band_batch']
    assert len(proof['checks']) == 2
    assert all(check['invalid_pixels_zero'] and check['odd_grid_preserved'] for check in proof['checks'])
