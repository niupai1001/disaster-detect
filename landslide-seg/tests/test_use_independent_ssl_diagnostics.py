from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_use_independent_ssl import view_statistics, identity_aliases, reconstruction_errors


def test_paired_and_nonpaired_views_are_distinguished():
    stats = view_statistics(torch.eye(4),torch.eye(4))
    assert stats['positive_cosine'] == 1 and stats['nonpaired_negative_cosine'] == 0
    assert stats['retrieval_recall']['1'] == 1
    with pytest.raises(ValueError): view_statistics(torch.ones(1,4),torch.ones(1,4))


def test_literal_band_aliases_and_standardized_error_are_explicit():
    proof = identity_aliases([dict(band_ids=['blue']),dict(band_ids=['B02'])])
    pair = proof['same_physical_band_different_names'][0]
    assert pair['random_identity_sha256'][0] != pair['random_identity_sha256'][1]
    values = dict(s2=dict(standardized_mae=.4,reflectance=dict(B02=dict(mae=.1,rmse=.2,pixels=10))))
    result = reconstruction_errors(values,dict(s2=dict(std=[.5])))
    assert result['s2']['bands']['B02']['standardized_rmse'] == .4
