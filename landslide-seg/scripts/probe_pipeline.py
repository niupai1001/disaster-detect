import sys, json, time, numpy as np, torch
sys.path.insert(0, r"N:/Users/yzh/Disater/landslide-seg/src")
from landslide_seg.config import load_config, set_seed
from landslide_seg.data.certified import read_manifest, filter_records, make_batches
from landslide_seg.models.build import build_from_config
from landslide_seg.encoders.terramind_tim import normalize_s2l2a_dn
from landslide_seg.data.band_mapping import normalize_anysat

set_seed(1234)
cfg = load_config(r"N:/Users/yzh/Disater/landslide-seg/configs/default.yaml")
recs = filter_records(read_manifest(r"N:/Users/yzh/Disater/landslide-seg/manifests/certified.jsonl"), split="test")
batch = make_batches(recs[:2], batch_size=2, shuffle=False)[0].to("cuda")
out = {}

x12, x10 = batch.x12, batch.x10
out["input"] = {
  "x12_shape": list(x12.shape), "x10_shape": list(x10.shape), "dates": batch.s2_dates.flatten().tolist(),
  "x12_dn_mean_per_band": [round(float(v),1) for v in x12.mean(dim=(0,2,3))],
  "x12_dn_std_per_band": [round(float(v),1) for v in x12.std(dim=(0,2,3))],
  "x10_dn_mean_per_band": [round(float(v),1) for v in x10.mean(dim=(0,2,3))],
  "valid_ratio": [round(float(v),4) for v in batch.valid.float().mean(dim=(1,2))],
  "mask_px": [int(v) for v in batch.mask.flatten(1).sum(1)],
}
xn = normalize_s2l2a_dn(x12)
out["input"]["x12_normalized_mean"] = [round(float(v),3) for v in xn.mean(dim=(0,2,3))]
out["input"]["x12_normalized_std"]  = [round(float(v),3) for v in xn.std(dim=(0,2,3))]

m = build_from_config(cfg, device="cuda"); m.eval()

# TerraMind
tm = m.terramind
t0=time.perf_counter(); r_on = tm(xn, tim=True); dt_on=time.perf_counter()-t0
t0=time.perf_counter(); r_off = tm(xn, tim=False); dt_off=time.perf_counter()-t0
tok_on, tok_off = r_on["tokens"][-1], r_off["tokens"][-1]
layers = r_on["tokens"]
out["terramind"] = {
  "grid": r_on["grid"], "tokens": list(tok_on.shape), "resampled_from": r_on["resampled_from"],
  "latency_tim_on_s": round(dt_on,3), "latency_tim_off_s": round(dt_off,3),
  "per_layer_std": [round(float(t.std()),4) for t in layers],
  "per_layer_mean": [round(float(t.mean()),4) for t in layers],
  "layer_cos_sim_adjacent": [round(float(torch.cosine_similarity(layers[i].flatten(), layers[i+1].flatten(), dim=0)),4) for i in range(len(layers)-1)],
  "tim_on_off_rel_diff": round(float((tok_on-tok_off).abs().mean()/tok_off.abs().mean()),4),
  "token_spatial_autocorr_lag1": round(float(np.corrcoef(tok_on[0,:-1,0].cpu(), tok_on[0,1:,0].cpu())[0,1]),4),
}
# AnySat
f40 = m.anysat(normalize_anysat(x10).unsqueeze(1), batch.s2_dates, patch_size_m=40)
f160 = m.anysat(normalize_anysat(x10).unsqueeze(1), batch.s2_dates, patch_size_m=160)
a40, a160 = f40["feature"], f160["feature"]
def blockiness(img, period):
    dx = img.abs().diff(dim=3).mean(dim=(0,1,2))
    idx = torch.arange(len(dx), device=dx.device)
    edge = ((idx+1) % period == 0)
    return float(dx[edge].mean()), float(dx[~edge].mean())
e4, i4 = blockiness(a40, 4); e16, i16 = blockiness(a160, 16)
out["anysat"] = {
  "dense_shape": list(a40.shape), "official_shape": list(f40["info"].out_shape),
  "num_patches_40m": f40["info"].num_patches, "num_patches_160m": f160["info"].num_patches,
  "mean": round(float(a40.mean()),4), "std": round(float(a40.std()),4),
  "per_channel_std_median": round(float(a40.std(dim=(0,2,3)).median()),4),
  "frac_channels_std_lt_001": round(float((a40.std(dim=(0,2,3))<0.01).float().mean()),4),
  "blockiness_40m_edge_vs_inner": [round(e4,5), round(i4,5)],
  "blockiness_160m_edge_vs_inner": [round(e16,5), round(i16,5)],
  "corr_40m_vs_160m": round(float(torch.cosine_similarity(a40.flatten(), a160.flatten(), dim=0)),4),
  "latency_40m_s": round(f40["info"].latency_s,3), "latency_160m_s": round(f160["info"].latency_s,3),
}
# semantic
sem = m.semantic
proto = sem.class_prototype
with torch.no_grad():
    o = m(x12, x10, batch.s2_dates, region_mask=batch.mask.unsqueeze(1).float(), valid=batch.valid, return_features=True)
sm = o.sem_map
gamma, beta = sem.film_params()
out["semantic"] = {
  "embed_dim": sem.embed_dim, "prototype_norm": round(float(proto.norm()),4),
  "attribute_prototypes_shape": list(sem.attribute_prototypes.shape),
  "attributes_active": sem.attributes_active,
  "sem_map_min": round(float(sm.min()),4), "sem_map_max": round(float(sm.max()),4),
  "sem_map_std": round(float(sm.std()),4),
  "semantic_scale": round(float(m.coarse.semantic_scale),4),
  "logit_scale": round(float(sem.logit_scale),4),
  "coarse_logit_range": round(float(o.coarse_logits.max()-o.coarse_logits.min()),4),
  "semantic_contrib_range": round(float((m.coarse.semantic_scale*sm).max()-(m.coarse.semantic_scale*sm).min()),4),
  "film_gamma_absmean": round(float(gamma.abs().mean()),4), "film_beta_absmean": round(float(beta.abs().mean()),4),
  "region_exists": o.region_exists.tolist(),
  "region_emb_norm": [round(float(v),4) for v in o.region_embedding.norm(dim=-1).flatten()],
  "region_cos_to_proto": [round(float(v),4) for v in torch.nn.functional.cosine_similarity(o.region_embedding.flatten(0,1), proto.expand(4,-1), dim=-1)],
}
# fusion
py = o.features["pyramid"]
out["fusion"] = {
  "source_channels": o.features["channels"],
  "pyramid_shapes": [list(p.shape) for p in py],
  "pyramid_std": [round(float(p.std()),4) for p in py],
  "gate": m.fusion.gate_report(),
}
print(json.dumps(out, ensure_ascii=False, indent=1))
open(r"N:/Users/yzh/Disater/landslide-seg/outputs/logs/pipeline_probe.json","w",encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
