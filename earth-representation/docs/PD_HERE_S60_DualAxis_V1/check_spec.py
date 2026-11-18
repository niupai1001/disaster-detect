"""Only arithmetic / pure-index checks. Not a neural-model or training test."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def block_mac(length: int, width: int) -> int:
    if length <= 0 or width <= 0:
        raise ValueError('length and width must be positive')
    return 12 * length * width * width + 2 * length * length * width


def main() -> None:
    checks = []
    sizes = [36,36,36,9,9,9,36,9,9,9]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    assert offsets == [0,36,72,108,117,126,135,171,180,189,198]
    checks.append('band_offsets_and_R198')

    assert 64*198 == 16*792 == 12672
    checks.append('total_native_value_count_preserved_for_same_480m_window')

    # Prove row-major patch extraction / raster assembly, including odd 3x3 blocks.
    for side in (6,3):
        width = side * 8
        image = [[row*width+col for col in range(width)] for row in range(width)]
        patches = [[image[py*side+dy][px*side+dx]
                    for dy in range(side) for dx in range(side)]
                   for py in range(8) for px in range(8)]
        rebuilt = [[patches[(y//side)*8+x//side][(y%side)*side+x%side]
                    for x in range(width)] for y in range(width)]
        assert rebuilt == image
        assert len(set(v for patch in patches for v in patch)) == width**2
    checks.append('6x6_and_3x3_exact_row_major_roundtrip_no_duplicate_pixels')

    old_keep = {0,5,10,15}
    new_keep = {y*8+x for y in range(8) for x in range(8)
                if (y//2)*4 + x//2 in old_keep}
    assert len(new_keep) == 16
    for fine_side in (6,3):
        width = fine_side*8
        old_side = fine_side*2
        old_mask = [(y//old_side)*4+x//old_side not in old_keep
                    for y in range(width) for x in range(width)]
        new_mask = [(y//fine_side)*8+x//fine_side not in new_keep
                    for y in range(width) for x in range(width)]
        assert old_mask == new_mask
    checks.append('old120m_mask_maps_to_identical_native_hidden_pixels_at60m')

    budget = json.loads((ROOT/'COMPUTE_BUDGET.json').read_text())
    for entry in budget['rows'].values():
        assert sum(entry['mac_by_component'].values()) == entry['forward_matrix_mac']
    mean_b = .4*budget['rows']['B1_hide_10m']['forward_matrix_mac'] + .6*budget['rows']['B1_hide_20m']['forward_matrix_mac']
    assert mean_b == budget['b1_uniform_band_mean_mac']
    assert 16*(mean_b+budget['rows']['A']['forward_matrix_mac']) == budget['batch16_paired_mac']
    checks.append('matrix_MAC_sums_and_batch16_paired_average')

    assert budget['rows']['A']['mac_by_component']['spatial_block'] == block_mac(16,128)
    assert budget['rows']['B1_hide_10m']['mac_by_component']['local_fusion_blocks'] == 64*block_mac(10,128)
    checks.append('A_and_B1_attention_scope_counts')

    result = {'status':'PASS_ARITHMETIC_AND_INDEX_ONLY','count':len(checks),'checks':checks,
              'neural_model_correctness':'NOT_RUN','hidden_value_model_isolation':'NOT_RUN',
              'training_quality':'NOT_RUN','GPU_timing':'NOT_RUN','DDP':'NOT_RUN'}
    (ROOT/'CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
