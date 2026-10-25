from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_use_independent import attention_counts


def test_attention_counts_include_padding_grid_heads_and_query_pool():
    result = attention_counts(3,5,7,dict(width=32,heads=4,queries=4,relation_grid=2))
    assert result['relation_positions'] == 12 and result['band_tokens'] == 36
    assert result['attention_scores_all_heads'] == 12*4*(3*3+4*3)
    assert result['qk_and_av_multiply_accumulates'] == 2*32*12*(3*3+4*3)
    mean = attention_counts(10,128,128,dict(attention=False,queries=0))
    assert mean['attention_scores_all_heads'] == 0
