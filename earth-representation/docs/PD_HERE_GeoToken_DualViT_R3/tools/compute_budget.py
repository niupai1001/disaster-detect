"""Analytical forward MAC budget. It is NOT a GPU timing benchmark."""
import json
from pathlib import Path


def block(length, width, mlp_ratio=4):
    return (4+2*mlp_ratio)*length*width**2+2*length**2*width


def budget(patches=16, shapes=((12,12),)*4+((6,6),)*6,
           d=128, final_width=128, dec=64, visible_patches=None, hidden_band=None):
    n = len(shapes)
    pixels = [h*w for h,w in shapes]
    m = [x//4 for x in pixels]
    pv = patches if visible_patches is None else visible_patches
    shown = [j for j in range(n) if j != hidden_band]
    nv = len(shown)
    # Separate linear QA embeddings double value-projection MACs.
    out = {
        'raw_and_qa_embeddings':pv*4*sum(pixels[j] for j in shown)*d,
        'spectral_block':pv*block(nv,d),
        'spatial_block':pv*sum(block(m[j],d) for j in shown),
        'spectral_compressor':pv*n*d*final_width,
        'spatial_compressor':pv*sum(m)*d*final_width,
        'branch_fusion':pv*2*final_width**2,
        'availability_to_encoder':pv*n*final_width,
        'cross_patch_block':block(pv,final_width),
        'encoder_to_decoder':pv*final_width*dec,
        'availability_to_decoder':patches*n*dec,
        'decoder_blocks':2*block(patches,dec),
        'native_output_projection':patches*dec*sum(pixels),
    }
    out['total_forward_mac'] = sum(out.values())
    out['geographic_encoder_tokens'] = pv
    out['decoder_tokens'] = patches
    out['visible_band_count'] = nv
    out['spatial_flatten_projection_weight_count'] = sum(m)*d*final_width
    return out


def main():
    a = budget(visible_patches=4)
    b = [budget(hidden_band=i) for i in range(10)]
    full = budget()
    avg_b = sum(x['total_forward_mac'] for x in b)/10
    result = dict(
        status='ANALYTICAL_NOT_MEASURED',
        assumptions=dict(patches=16,band_count=10,native_shapes=[[12,12]]*4+[[6,6]]*6,
                         branch_width=128,geo_width=128,decoder_width=64,decoder_depth=2,
                         qa_projection=True),
        scope='Forward linear and matmul MACs only. Excludes normalization, nonlinearities, softmax, '
              'backward, optimizer, padding, data transfer, CPU dispatch and synchronization.',
        no_mask=full,A_75_percent=a,B1_by_hidden_band=b,
        B1_uniform_band_mean_forward_mac=avg_b,
        paired_global16_forward_mac=16*(a['total_forward_mac']+avg_b),
        comparisons=dict(
            old_all_fine_tokens_global_fusion_encoder_transformer_mac=4167606272,
            coarse_16token_vit_two_blocks_mac=2*block(16,128),
            warning='Comparisons describe different architectures/capacities, not equal-quality speedups.'),
    )
    destination = Path(__file__).resolve().parents[1]/'04_COMPUTE_BUDGET.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','B1_uniform_band_mean_forward_mac','paired_global16_forward_mac')},indent=2))
    for k in ('no_mask','A_75_percent'):
        print(k,result[k]['total_forward_mac']/1e9,'G MAC')

if __name__ == '__main__':
    main()
