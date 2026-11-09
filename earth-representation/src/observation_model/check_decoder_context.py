"""G2 large-hole decoder check; no additional CUDA context or optimizer."""
import torch

from .encoder_v2 import ObservationEncoderV2,OpticalImage,target_from_band
from .check_encoder_v2 import DEST,fixed_images,masked_rectangle,state_hash
from .expansion_data import save
from dataclasses import replace


def main():
    torch.set_num_threads(4)
    model=ObservationEncoderV2().eval()
    model.load_state_dict(torch.load(DEST/'initialization_untrained.pt',map_location='cpu',weights_only=True)['model'])
    before=state_hash(model);images,_=fixed_images();image=images[0];target=target_from_band(image.bands[0])
    with torch.no_grad():
        reference=torch.load(DEST/'native_prediction_before_context_fix.pt',map_location='cpu',weights_only=True)
        native,q=model.predict(image,target)
        assert torch.equal(native,reference['prediction']) and torch.equal(q,reference['valid'])
        base=model.encode(image);left,bottom,right,top=base.grids[0]['bounds']
        cx,cy=(left+right)/2,(bottom+top)/2
        masked=masked_rectangle(image,(cx-300,cy-300,cx+300,cy+300))
        rep=model.encode(masked);ny,nx=rep.grids[0]['shape'];valid=rep.valid[0].reshape(ny,nx)
        assert not valid[ny//2-1:ny//2+2,nx//2-1:nx//2+2].any()
        prediction,coverage=model.decoder(rep,target)
        assert coverage[target.shape[0]//2,target.shape[1]//2]
        altered=OpticalImage([replace(b,values=torch.where(b.visible,b.values,b.values+10000),
            valid=torch.where(b.visible,b.valid,~b.valid)) for b in masked.bands])
        other,other_q=model.predict(altered,target)
        assert torch.equal(prediction,other) and torch.equal(coverage,other_q)
    band=replace(masked.bands[0],values=masked.bands[0].values.clone().requires_grad_(True))
    prediction,coverage=model.predict(OpticalImage([band]+masked.bands[1:]),target)
    hidden=(~band.visible)&band.valid&coverage
    loss=(prediction[hidden]-image.bands[0].values[hidden]).square().mean();loss.backward()
    assert torch.isfinite(loss) and band.values.grad[~band.visible].eq(0).all()
    assert band.values.grad[band.visible].abs().sum()>0
    assert before==state_hash(model)
    save(DEST/'decoder_context_tests.json',dict(status='PASS',large_hole_m=600,
        no_local_3x3_key_at_target_center=True,large_hole_prediction_available=True,
        hidden_value_QA_intervention_exact_zero=True,hidden_input_gradients_exact_zero=True,
        visible_context_gradients_nonzero=True,native_GPU_profile_case_CPU_outputs_bitwise_unchanged=True,
        profile_scope='One original GPU profile preserved; final global-H fallback checked on CPU only. No extra CUDA context.',
        model_hash_before=before,model_hash_after=state_hash(model),optimizer_updates=0))
    print('G2_LARGE_HOLE_CONTEXT_PASS')


if __name__=='__main__':main()
