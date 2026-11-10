"""Shared projections and packed backward versus an explicit, unbatched attention reference."""
import unittest
import torch
from encoder_v3.latent_response.data import make_view,target_chunks
from encoder_v3.latent_response.decoder import exact_knn
from encoder_v3.latent_response.geometry import fourier2d
from encoder_v3.latent_response.execution import view_loss
from tests.encoder_v3.latent_response.tests.test_pipeline import fixture


def direct_prediction(model,rep,channel,points):
    decoder=model.decoder;geometry=rep.geometry
    condition=model.condition('decoder',rep.sensor_key,channel.channel_meta,channel.gsd_xy,rep.memory)
    patch=torch.floor((points.double()-points.new_tensor(geometry.origin))/geometry.patch_size)
    center=(patch+.5)*geometry.patch_size+points.new_tensor(geometry.origin)
    relative=((points-center)/geometry.patch_size).to(condition.dtype)
    gsd=channel.gsd_xy
    support=condition.new_tensor([gsd[0]/geometry.patch_size,gsd[1]/geometry.patch_size,gsd[0]*gsd[1]/geometry.patch_size**2]).log()
    geo=torch.cat((fourier2d(relative),support.expand(len(points),-1)),-1)
    z=condition.expand(len(points),-1);query=decoder.query(torch.cat((z,geo),-1))[:,None]
    selected,_=exact_knn(points,rep.centers)
    position=decoder.relative(fourier2d(((rep.centers[selected]-points[:,None])/geometry.patch_size).to(query.dtype)))
    summary=rep.fine[selected]+position+decoder.types[0]
    latent=rep.latent[selected]+position[:,:,None]+decoder.types[1]
    kv=torch.cat((summary[:,:,None],latent),2).flatten(1,2)
    available=torch.cat((torch.ones_like(rep.latent_valid[selected][...,:1]),rep.latent_valid[selected]),-1).flatten(1)
    local,_=decoder.local(query,kv,kv,key_padding_mask=~available,need_weights=False)
    shared=rep.global_latent[None].expand(len(points),-1,-1)
    global_read,_=decoder.global_read(query,shared,shared,need_weights=False)
    has_latent=rep.latent_valid[selected].any(-1).any(-1).to(query.dtype)[:,None]
    joined=torch.cat((local[:,0],global_read[:,0],geo,z,torch.ones_like(has_latent),has_latent),-1)
    return decoder.output(joined).squeeze(-1)


class ProjectedDecoderTest(unittest.TestCase):
    def test_full_loss_predictions_and_all_gradients_match_direct_attention(self):
        torch.set_num_threads(2)
        for kind in ('A','B1'):
            current,cs,obs=fixture(16);reference,_,_=fixture(16)
            reference.load_state_dict(current.state_dict());view=make_view(obs,cs,kind)
            predictions=[]
            hook=current.decoder.register_forward_hook(lambda m,a,o:predictions.append(o[0].detach().clone()))
            result=view_loss(current,view,backward=True,chunk_pixels=37);hook.remove()
            rep=reference.encode(view.channels,view.geometry)
            expected=[];loss=0
            for channel,mask,truth in view.targets:
                points,y=next(target_chunks(channel,mask,truth,10000))
                prediction=direct_prediction(reference,rep,channel,points)
                expected.append(prediction.detach());loss+=(prediction-y).square().mean()/len(view.targets)
            loss.backward()
            torch.testing.assert_close(torch.cat(predictions),torch.cat(expected),atol=1e-6,rtol=1e-5)
            self.assertAlmostEqual(result['loss'],float(loss.detach()),places=6)
            wanted={n:p.grad for n,p in reference.named_parameters() if p.grad is not None}
            actual={n:p.grad for n,p in current.named_parameters() if p.grad is not None}
            self.assertEqual(actual.keys(),wanted.keys())
            for name,gradient in wanted.items():
                torch.testing.assert_close(gradient,actual[name],atol=1e-5,rtol=1e-4,msg=name)
