"""Offline official AnySat; reject inputs outside the documented product contract."""
from datetime import datetime
import math

import torch
from torch import nn

from landslide_seg.encoders.anysat_scale import _load_hubconf
from landslide_seg.data.band_mapping import normalize_anysat
from landslide_seg.models.dofa import _module


S2_BANDS=('blue','green','red','rededge1','rededge2','rededge3','nir','nir08','swir16','swir22')


class NativeDOFA(nn.Module):
    """Complete published encoder checkpoint with official dense inference forward.

    The released 173-key checkpoint is encoder-only, not the MAE training state.
    Retain its norm/projector/mask parameters even though dense forward does not
    consume them; no key is discarded and the official patch/trunk are unchanged.
    """
    def __init__(self,root):
        super().__init__()
        import sys
        for path in (root/'vendor/dofa-official-0cfb7e1',root/'vendor/dofa_runtime'):
            if str(path) not in sys.path:sys.path.insert(0,str(path))
        source=root/'vendor/dofa-official-0cfb7e1'
        official=_module('use_next_native_dofa_dense',source/'downstream_tasks/geobench_segmentation/model.py')
        self.official=official.vit_base_patch16(img_size=224,drop_path_rate=0.)
        self.official.norm=nn.LayerNorm(768,eps=1e-6)
        self.official.projector=nn.Linear(768,768)
        self.official.mask_token=nn.Parameter(torch.zeros(1,1,512))
        checkpoint=torch.load(root/'weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth',map_location='cpu',weights_only=True)
        state=checkpoint.get('model',checkpoint)
        self.official.load_state_dict(state,strict=True)
        self.loaded_keys=tuple(state)
        self.requires_grad_(False); self.eval()

    def train(self,mode=True):return super().train(False)

    def prepare(self,asset):
        asset.validate()
        if asset.modality not in ('s2','landsat') or asset.radiometry!='surface_reflectance':
            raise ValueError('DOFA optical reflectance input required')
        if not asset.metadata_known[:,0].all(): raise ValueError('DOFA requires real known wavelengths')
        x=torch.where(asset.pixel_valid,asset.image,0)[None]
        if x.shape[-2:]!=(224,224):
            # Same physical extent, explicit interpolation for legal native 224 inference.
            x=torch.nn.functional.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
        return x,asset.channel_metadata[:,0].tolist()

    @torch.no_grad()
    def forward(self,asset):return self.official(*self.prepare(asset))


class NativeAnySat(nn.Module):
    def __init__(self,root):
        super().__init__()
        self.official=_load_hubconf(root/'vendor/AnySat-main').AnySat(model_size='base',flash_attn=False)
        checkpoint=torch.load(root/'weights/AnySat_base.pth',map_location='cpu',weights_only=False)
        self.official.model.load_state_dict(checkpoint['state_dict'],strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self,mode=True):
        return super().train(False)

    def prepare(self,asset,hidden_pixels=None):
        asset.validate()
        if asset.modality!='s2' or asset.radiometry!='surface_reflectance':
            raise ValueError('Native AnySat product not qualified: only S2 surface reflectance implemented')
        if set(asset.channel_names)!=set(S2_BANDS):
            raise ValueError('AnySat requires exactly the official ten real S2 bands')
        a,b,_,d,e,_=asset.sample_grid['transform']
        if abs(math.hypot(a,d)-10)>1e-5 or abs(math.hypot(b,e)-10)>1e-5 or abs(a*b+d*e)>1e-5:
            raise ValueError('AnySat requires orthogonal 10m sample axes')
        if asset.image.shape[-2]!=asset.image.shape[-1] or asset.image.shape[-1]%4:
            raise ValueError('Native AnySat square input must be divisible by 4 for 40m patches')
        indices=[asset.channel_names.index(band) for band in S2_BANDS]
        valid=asset.pixel_valid[indices]
        x=normalize_anysat(torch.where(valid,asset.image[indices],0)*10000)
        visible=valid.clone()
        if hidden_pixels is not None:
            if hidden_pixels.shape!=x.shape[-2:]: raise ValueError('Mask grid mismatch')
            visible &= ~hidden_pixels[None]
        x=torch.where(visible,x,0)
        date=datetime.fromisoformat(asset.acquisition_time.replace('Z','+00:00'))
        day=date.timetuple().tm_yday-1
        return {'s2':x[None,None], 's2_dates':torch.tensor([[day]],device=x.device)}

    @torch.no_grad()
    def forward(self,asset,hidden_pixels=None):
        return self.official(self.prepare(asset,hidden_pixels),patch_size=40,output='dense',output_modality='s2')
