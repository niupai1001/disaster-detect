"""One mixed native-grid batch covers query, permutation, masking and gradients."""
import json
import os
from pathlib import Path
import torch

if __name__=="__main__" and torch.cuda.is_available():torch.cuda.init()

from .model import ObservationModel,BandRaster,TargetQuery


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model=ObservationModel(120,coordinate_mode=os.environ.get("OBSERVATION_COORDINATE_MODE","legacy_local")).to(device)
    bands=[]
    for name,g,wave in [("S2:B02",10,492.),("S2:B05",20,704.),("L8:SR_B5",30,865.)]:
        value=torch.rand(2,120//g,120//g,device=device,requires_grad=True)
        valid=torch.ones_like(value,dtype=torch.bool);valid[0,:2,:2]=False
        bands.append(BandRaster(name,value,valid,wave,None,g,(g/2,g/2)))
    query=TargetQuery("S2","S2:B02",492.,None,10,10)
    pred,valid=model.predict_bands(bands,query)
    swapped,_=model.predict_bands(list(reversed(bands)),query)
    torch.testing.assert_close(pred,swapped,rtol=2e-5,atol=2e-6)
    original=bands[0].value;bands[0].value=torch.full_like(original,999.)
    hidden,_=model.predict_bands(bands,query)
    torch.testing.assert_close(pred,hidden)
    bands[0].value=original
    representation=model.encode(bands[1:])
    subset=torch.tensor([0,3],device=device)
    sparse,_=model.decoder(representation,query,subset)
    expected=pred.reshape(2,2,6,2,6).permute(0,1,3,2,4).reshape(2,4,36)[:,subset]
    torch.testing.assert_close(sparse,expected,rtol=2e-5,atol=2e-6)
    changed_query=TargetQuery("S2","S2:B11",1614.,None,20,10)
    changed,_=model.decoder(representation,changed_query)
    assert float((changed-pred).abs().max().detach())>1e-6
    # A valid native10m impulse must reach the encoded representation.
    value=original.detach().clone();single=BandRaster("S2:B02",value,bands[0].valid,492.,None,10,(5,5))
    first=model.encode([single]).tokens.detach();single.value=value.clone();single.value[:,3,3]+=.01
    assert float((model.encode([single]).tokens-first).abs().max().detach())>1e-6
    other,_=model.predict_bands(bands,TargetQuery("S2","S2:B11",1614.,None,20,20))
    assert other.shape==(2,6,6) and pred.shape==(2,12,12)
    pred.square().mean().backward()
    assert original.grad is None
    assert bands[1].value.grad is not None and bands[1].value.grad.isfinite().all()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    out=Path(os.environ.get("OBSERVATION_CHECK_DIR",Path(__file__).parent/"outputs/phase1"));out.mkdir(parents=True,exist_ok=True)
    result=dict(status="synthetic_mixed_batch_pass_only",device=device,parameters=sum(p.numel() for p in model.parameters()),permutation_max_error=float((pred-swapped).abs().max().detach()),hidden_target_zero_influence=True,gradients_finite=True,native_data_acceptance="pending",formal_updates=0)
    (out/"model_check.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
