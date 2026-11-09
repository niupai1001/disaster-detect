"""A constructive spatial-information counterexample on a native10m grid."""
import importlib
import sys
import types

import numpy as np
import torch

from .encoder_v2_runtime import BASE
from .expansion_data import read,save


def main():
    torch.set_num_threads(4);dest=BASE/'pre_G5/information_check';dest.mkdir(exist_ok=True)
    package=types.ModuleType('_g4_information_check');package.__path__=[str(BASE/'G4_fit/source')];sys.modules[package.__name__]=package
    m=importlib.import_module(package.__name__+'.encoder_v2')
    manifest=read(BASE/'G1/srf_manifest.json')['curves']
    curves={}
    for band in ['B03','B08']:
        c=next(c for c in manifest if c['instrument']=='S2A' and c['band_key']==band and c['version']=='4.0')
        with np.load(c['path']) as z:curves[band]=m.srf_descriptor(z['spectral_wavelength_grid_nm'],z['normalized_response'])
    yy,xx=torch.meshgrid((torch.arange(6)+.5)*10/1200,(torch.arange(6)+.5)*10/1200,indexing='ij')
    matrix=torch.stack([torch.ones_like(xx),xx,yy,xx*xx,yy*yy,xx*yy,
        torch.sin(2*torch.pi*xx),torch.cos(2*torch.pi*xx),torch.sin(2*torch.pi*yy),torch.cos(2*torch.pi*yy)]).reshape(10,36).double()
    _,singular,vh=torch.linalg.svd(matrix,full_matrices=True)
    delta=vh[-1].reshape(6,6);delta=(delta/delta.abs().max()*.02).float()
    first=torch.full((120,120),.2);second=first.clone();second[:6,:6]+=delta
    a=m.OpticalBand(first,torch.ones_like(first,dtype=torch.bool),(10.,0.,500000.,0.,-10.,4000000.),'EPSG:32650',curves['B03'],'B03')
    from dataclasses import replace
    b=replace(a,values=second)
    model=m.ObservationEncoderV2().eval()
    model.load_state_dict(torch.load(BASE/'G4_fit/best.pt',map_location='cpu',weights_only=False)['model'])
    target=m.TargetGrid((120,120),a.transform,a.crs,curves['B08'],'B08')
    with torch.no_grad():
        h1=model.encode(m.OpticalImage([a]));h2=model.encode(m.OpticalImage([b]))
        y1,_=model.decoder(h1,target);y2,_=model.decoder(h2,target)
    change=float((h1.features-h2.features).abs().max());prediction_change=float((y1-y2).abs().max())
    save(dest/'original.json',dict(status='INFORMATION_LOSS_DEMONSTRATED',native_input_pixels_per_cell=36,linear_moments=10,
        minimum_nullspace_dimension=26,matrix_delta_max=float((matrix@delta.flatten().double()).abs().max()),
        input_max_change=float((first-second).abs().max()),encoder_H_max_change=change,decoder_prediction_max_change=prediction_change,
        synthetic_linear_target_max_change=float((2*first-2*second).abs().max()),
        caveat='Constructed physically plausible values on a native10m geometry, not a claim about a measured target relation. This proves lost fine information can exist; not the sole cause of G4 failure.',
        optimizer_updates=0))
    np.savez_compressed(dest/'counterexample.npz',first=first.numpy(),second=second.numpy(),delta=delta.numpy(),
        prediction_first=y1.numpy(),prediction_second=y2.numpy())
    print('INFORMATION_CHECK',change,prediction_change,float((first-second).abs().max()))


if __name__=='__main__':main()
