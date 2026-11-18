"""Read-only checkpoint interventions and bounded R3 synthetic fitting."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn

from train_pd_here_geotoken_dualvit_r3 import model_for, update, load_panel
from audit_pd_here_r3_sources import BASE, read
from pd_here_geotoken_dualvit_r3.data import geometry, masks, load_window, batch, OpticalDataset, write_json
from pd_here_geotoken_dualvit_r3.model import masked_reconstruction_loss, GeoTokenDualViT
from train_pd_here_r3_texture_study import initial_model, TASKS


def rms(x):
    return x.double().square().mean().sqrt().item()


@torch.no_grad()
def intervention(model, values, qa, view, task):
    keep,bands = view
    active = [q & keep[:,:,None,None] & bands[:,j,None,None,None] for j,q in enumerate(qa)]
    means = [torch.where(a,x,0.).flatten(1).sum(-1)/a.flatten(1).sum(-1).clamp_min(1)
             for x,a in zip(values,active,strict=True)]
    flattened = [torch.where(a,m[:,None,None,None],x) for x,a,m in zip(values,active,means,strict=True)]
    generator = torch.Generator(device=keep.device).manual_seed(17)
    shuffled = []
    for x,a in zip(values,active,strict=True):
        candidate = x.clone()
        for i,j in a.flatten(2).any(-1).nonzero().tolist():
            valid = a[i,j]
            pixels = x[i,j][valid]
            candidate[i,j][valid] = pixels[torch.randperm(len(pixels),device=keep.device,generator=generator)]
        shuffled.append(candidate)
    hidden = [torch.where(a,x,torch.full_like(x,99.)) for x,a in zip(values,active,strict=True)]
    names = ('spatial_values','spectral_values.12x12','spectral_values.6x6',
             'spectral_reduce','spatial_reduce','encoder_norm','decoder_norm','prediction')
    measurements = {}
    modules = dict(model.named_modules())
    names = tuple(name for name in names if name in modules and not isinstance(modules[name], nn.ModuleDict))
    hooks = [modules[name].register_forward_hook(
        lambda m,args,result,name=name: measurements.setdefault(name,[]).append(result))
        for name in names]
    try:
        original = model(values,qa,*view)
        base = dict(measurements)
        effects = {}
        for label,data in [('mean',flattened),('shuffle',shuffled),('hidden',hidden)]:
            measurements.clear()
            model(tuple(data),qa,*view)
            effects[label] = {name:rms(torch.cat([(x-y).flatten() for x,y in zip(parts,measurements[name],strict=True)]))
                              for name,parts in base.items()}
        if any(value != 0 for value in effects['hidden'].values()):
            raise ValueError('Hidden values changed learned features')
    finally:
        for hook in hooks:
            hook.remove()
    features = base['decoder_norm'][0]
    prediction = original['prediction']
    if features.shape[1] != prediction.shape[1]:
        features = features.reshape(features.shape[0],prediction.shape[1],len(values),features.shape[-1])
    offset = 0
    examples = []
    for j,(y,q) in enumerate(zip(values,qa,strict=True)):
        pixels = y.shape[-2]*y.shape[-1]
        p = prediction[...,offset:offset+pixels]
        offset += pixels
        target = q.flatten(2).all(-1) & ((~keep) if task=='A' else ~bands[:,j,None])
        for i in range(len(y)):
            if target[i].sum()<2:
                continue
            pp = p[i,target[i]]
            yy = y[i,target[i]].flatten(1)
            pp = pp-pp.mean(-1,keepdim=True)
            yy = yy-yy.mean(-1,keepdim=True)
            f = torch.nn.functional.normalize(features[i,target[i],j] if features.ndim == 4 else features[i,target[i]],dim=-1)
            similarity = (f@f.T)[~torch.eye(len(f),device=f.device,dtype=torch.bool)]
            examples.append(dict(parent=i,band=j,prediction_texture_rms=rms(pp),truth_texture_rms=rms(yy),
                template_fraction=float(pp.mean(0).square().mean()/pp.square().mean().clamp_min(1e-20)),
                truth_template_fraction=float(yy.mean(0).square().mean()/yy.square().mean().clamp_min(1e-20)),
                common_texture_error_rms=rms((pp-yy).mean(0)),
                texture_error_rms=rms(pp-yy),
                cosine=float(similarity.mean())))
    return dict(layer_difference_rms=effects,texture=examples)


@torch.no_grad()
def affine_oracle(model, values, qa, views):
    weight = model.prediction.weight.double()
    bias = model.prediction.bias.double()
    truth = torch.cat([v.flatten(2) for v in values],-1).double()
    result = {}
    for task,(keep,bands) in zip(TASKS,views,strict=True):
        selected = [q.flatten(2) & ((~keep)[:,:,None] if task=='A' else (~bands[:,j])[:,None,None])
                    for j,q in enumerate(qa)]
        counts = [q.sum((1,2)).clamp_min(1) for q in selected]
        weights = torch.cat([q/count[:,None,None] for q,count in zip(selected,counts,strict=True)],-1).double()
        choices = weights.flatten(0,1)
        response = (truth-bias).flatten(0,1)
        legal = choices.any(-1)
        square_root = choices[legal].sqrt()
        system = weight[None]*square_root[...,None]
        right = response[legal]*square_root
        solution = torch.linalg.pinv(system,rtol=1e-8)@right[...,None]
        fit = (solution.squeeze(-1)@weight.T)+bias
        predicted = torch.zeros_like(truth)
        predicted.flatten(0,1)[legal] = fit
        result[task] = masked_reconstruction_loss(predicted,values,qa,keep,bands,task).item()
    return result


def easy_fit(layout, task, device):
    model,opt = model_for(layout,device)
    gen = torch.Generator(device=device).manual_seed(17)
    # One common physical plane per parent. Every band is an affine transform of it.
    coefficients = torch.rand(16,3,device=device,generator=gen)
    tiles = torch.arange(16,device=device)
    values = []
    for j,(h,w) in enumerate(layout.native_shapes):
        x = (tiles[:,None,None]%4+(torch.arange(w,device=device)[None,None,:]+.5)/w)/4
        y = (tiles[:,None,None]//4+(torch.arange(h,device=device)[None,:,None]+.5)/h)/4
        c = coefficients[:,None,None,None]
        slope = .10 if task=='B1' else .04
        plane = .08 + c[...,0] * .13 + c[...,1]*slope*x + c[...,2]*slope*y
        values.append(plane*(1+j*.05)+j*.008)
    values = tuple(values)
    qa = tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    ids = [f'plane_{i}' for i in range(16)]
    view_index = TASKS.index(task)
    fixed = masks(qa,ids,[0]*16,0)[view_index]
    def score():
        with torch.no_grad():
            return masked_reconstruction_loss(model(values,qa,*fixed)['prediction'],values,qa,*fixed,task).item()
    before = score()
    for step in range(200):
        update(model,opt,values,qa,masks(qa,ids,[0]*16,step),task)
    return dict(initial_mse=before,final_mse=score(),updates=200)


def fit_decoder(layout,values,qa,views,device,mode,path):
    model,opt = model_for(layout,device)
    # Both arms start from the same trained encoder and decoder weights.
    model.load_state_dict(torch.load(BASE/'subset700_10epochs/last.pt',map_location=device,weights_only=True)['model'])
    with torch.no_grad():
        outputs = [model(values,qa,*view) for view in views]
    latents = [nn.Parameter(item['encoded'].detach().clone(),requires_grad=mode=='free') for item in outputs]
    opt.add_param_group(dict(params=[p for p in latents if p.requires_grad],weight_decay=0.))
    valid_geo = torch.stack([q.flatten(2).any(-1) for q in qa],-1).any(-1)
    def scores():
        losses = []
        for task,view,encoding,latent in zip(TASKS,views,outputs,latents,strict=True):
            pred,_,_ = model.decode(latent,encoding['ids_keep'],encoding['valid'],view[1],valid_geo)
            losses.append(masked_reconstruction_loss(pred,values,qa,*view,task))
        return torch.stack(losses)
    with torch.no_grad():
        initial = scores().tolist()
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        scores().mean().backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(),*latents],1.,error_if_nonfinite=True)
        opt.step()
    with torch.no_grad():
        final = scores().tolist()
    torch.save(dict(model=model.state_dict(),latents=[x.detach().cpu() for x in latents],
                    purpose='DIAGNOSTIC ONLY',mode=mode),path)
    return dict(initial_mse=dict(zip(TASKS,initial)),final_mse=dict(zip(TASKS,final)),updates=500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path,default=BASE/'texture_study')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=True)
    layout,values,qa,ids,windows = load_panel(BASE,args.device)
    views = masks(qa,ids,windows,0)
    result = {'scope':'synthetic and historical cache diagnostics, separate from audited comparison','content':{}}
    for name,file in [('init',None),('step200',BASE/'paired_last.pt'),('step640',BASE/'subset700_10epochs/last.pt')]:
        model,_ = model_for(layout,args.device)
        if file:
            model.load_state_dict(torch.load(file,map_location=args.device,weights_only=True)['model'])
        model.eval()
        result['content'][name] = {task:intervention(model,values,qa,view,task)
                                   for task,view in zip(TASKS,views,strict=True)}
    result['affine_oracle'] = {}
    for width in (64,128):
        torch.manual_seed(17)
        oracle_model = GeoTokenDualViT(layout,decoder_width=width).to(args.device)
        result['affine_oracle'][str(width)] = affine_oracle(oracle_model,values,qa,views)
    result['affine_oracle_scope'] = 'Seed-17 initialized heads, QA/task-weighted projection; no latent or LayerNorm constraints'
    rendered = read(BASE/'subset700_10epochs/predictions_val/examples.json')['examples']
    manifest = {row['id']:row for row in map(json.loads,(BASE/'val.jsonl').read_text().splitlines())}
    fixed = read(BASE/'window_plan.json')
    dataset = OpticalDataset(BASE.parents[1])
    result['shown_val'] = {}
    for task,examples in rendered.items():
        rows = [manifest[e['id']] for e in examples]
        v,q = batch([load_window(dataset,row,fixed[row['id']]['origin'],layout) for row in rows],args.device)
        view = masks(q,[row['id'] for row in rows],[e['window_id'] for e in examples],0)[TASKS.index(task)]
        result['shown_val'][task] = intervention(model,v,q,view,task)
    write_json(args.out/'diagnostics.json',result)
    result['easy_fit'] = {task:easy_fit(layout,task,args.device) for task in TASKS}
    write_json(args.out/'diagnostics.json',result)
    for mode in ('frozen','free'):
        result.setdefault('decoder_fit',{})[mode] = fit_decoder(
            layout,values,qa,views,args.device,mode,args.out/f'diagnostic_{mode}.pt')
        write_json(args.out/'diagnostics.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('content',)},indent=2),flush=True)


if __name__ == '__main__':
    main()
