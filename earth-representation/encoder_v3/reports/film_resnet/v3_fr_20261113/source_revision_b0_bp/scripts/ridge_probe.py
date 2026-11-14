"""Equal-width native target probes of the best replayable full-visible candidate."""
import csv

import numpy as np
import torch
from torch.nn import functional as F

from candidate_gate import config_device, options
from run_v3_fr import read, write


@torch.no_grad()
def run(args):
    from encoder_v3.film_resnet.frontend import pack
    from encoder_v3.film_resnet.model import FRPretraining,split_field
    from encoder_v3.hierarchical_revised.train import resume
    from encoder_v3.shared_mae.data import load_data,sample_from_dataset
    from encoder_v3.shared_mae.geometry import AreaCache

    root = args.output/"s5_capacity_fit"
    gate = read(root/"fit_gate.json")
    assert gate["status"]=="FIT_RUNS_VALID"
    cfg,_,device = config_device(args)
    model = FRPretraining(cfg,options("C"),seed=17).to(device)
    model.legacy.requires_grad_(False)
    bundle = torch.nn.ModuleDict({"model":model,"latents":torch.nn.ParameterDict()})
    opt = torch.optim.AdamW([p for p in bundle.parameters() if p.requires_grad],lr=.0003,weight_decay=0.)
    binding = read(root/"AE-F/identity.json")
    resume(root/"AE-F/best.pt",bundle,opt,{},binding)
    bundle.eval()
    data,resolver = load_data()
    panel = read(args.output/"data_panel.json")
    area = AreaCache(limit_bytes=64*1024**2)
    fusion = model.encoder.spectral
    names = ("lift","resnet","phase","band_slot","fused_slot","fine_backbone","decoder_parent","decoder_fine")
    features, identity = {}, {}
    for role in ("fit","read_only"):
        for item in panel[role]:
            s = sample_from_dataset(data["TRAIN"],item["index"],resolver,device)
            keep = [c.valid for c in s.channels]
            bands,encoded,_ = model.views(s,keep,keep)
            i = next(i for i,b in enumerate(bands) if b.visible.any())
            band = bands[i]
            p = pack(band)
            h = fusion.frontend.lift(p.values[None],p.visible[None],p.structural[None])
            hidden = fusion.frontend([p],encoded.descriptor[i:i+1])
            phase = fusion.phase_features(p,hidden)[0]
            r = p.factor
            def native_phase(field):
                y,x = torch.meshgrid(torch.arange(band.grid.shape[0],device=device),
                                     torch.arange(band.grid.shape[1],device=device),indexing="ij")
                j = (y%r)*r+x%r
                return field[j,:,y//r,x//r].reshape(-1,8)
            mapping = area.get(band.grid,encoded.geometry,2,device)
            value,_ = fusion.aggregate(band.grid,encoded.geometry,mapping,[band],keep[i][None],
                                      encoded.descriptor[i:i+1],262144)
            slots = fusion(bands,encoded.descriptor,encoded.geometry)
            def native_area(field,mapping):
                return mapping.native_predictions(field.T[:,None,:]).T
            def to_parent(field):
                return native_area(field[0,:8].flatten(1).T,area.get(band.grid,encoded.geometry,1,device))
            # Use eight features per native position; phase readouts have equal bandwidth.
            group = (h[0,:8].expand(r*r,-1,-1,-1),hidden[0,:8].expand(r*r,-1,-1,-1))
            columns = [native_phase(f) for f in (*group,phase)]
            columns += [native_area(value[0],mapping),native_area(slots.values.flatten(0,1),mapping),
                        to_parent(encoded.features[0]),to_parent(model.decoder.parent_field(encoded)),
                        native_area(model.decoder.subcells(encoded)[0].T,
                                    area.get(band.grid,encoded.geometry,4,device))]
            mask = keep[i].flatten()
            target = s.channels[i].values.flatten()[mask].double().cpu().numpy()
            features[s.identifier] = ({n:f[mask].double().cpu().numpy() for n,f in zip(names,columns,strict=True)},target)
            identity[s.identifier] = (role,item["sensor"],s.channels[i].name)
    rows = []
    for name in names:
        fit_ids = [item["id"] for item in panel["fit"]]
        fit = np.concatenate([features[id_][0][name] for id_ in fit_ids])
        truth = np.concatenate([features[id_][1] for id_ in fit_ids])
        mu,std = fit.mean(0),fit.std(0)
        std[std<1e-9] = 1
        x = (fit-mu)/std
        bias = truth.mean()
        # Use per-target mean square and fixed ridge strength.
        w = np.linalg.solve(x.T@x/len(x)+.001*np.eye(x.shape[1]),x.T@(truth-bias)/len(x))
        for id_,(by_layer,target) in features.items():
            role,sensor,band = identity[id_]
            pred = ((by_layer[name]-mu)/std)@w+bias
            rows.append(dict(interface=name,id=id_,role=role,sensor=sensor,band=band,native_targets=len(target),
                             fit_feature_rank=int(np.linalg.matrix_rank(x)),features=x.shape[1],
                             readout_parameters=x.shape[1]+1,mse=float(np.mean((pred-target)**2)),
                             zero_mse=float(np.mean(target**2))))
    with (root/"ridge_probe.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,rows[0])
        writer.writeheader()
        writer.writerows(rows)
    gate["ridge_probe_status"] = "PASS"
    gate["ridge_probe_limits"] = ("Eight fixed feature channels per interface, native footprint projection and one "
                                  "shared readout fitted only on the two TRAIN fit members. The lift/ResNet subset "
                                  "can omit phase carrier dimensions; probe failure is not loss of information proof.")
    gate["status"] = "PASS" if gate["effect_screened"] else "INCONCLUSIVE"
    write(root/"fit_gate.json",gate)
    print("S5 ridge probes",len(rows),"rows;",gate["status"],flush=True)
