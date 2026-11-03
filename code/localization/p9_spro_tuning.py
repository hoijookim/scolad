#!/usr/bin/env python3.12
"""
SALAD-Free Phase 9 — AU-sPRO 튜닝 (pixel-fusion localization)
=============================================================
현 Hybrid localizer = DINOv2-B L11 단일 per-position Maha → sPRO 0.394.
튜닝: DINOv3-L multi-layer(L8+L17+L23) per-position Maha + PC patch-min-dist + 기존 Hybrid 융합.
목표: LOCO 5-cat AU-sPRO 0.394 → 개선. leakage-free(per-position Maha·bank train만).
eval: src/spro_evaluation.evaluate_object @0.05.
"""
import numpy as np, glob, os, json, math, warnings, sys
from pathlib import Path
from PIL import Image
import cv2, torch
R=Path("/workspace/ai-vision-research"); sys.path.insert(0,str(R))
CACHE=R/"cache/dinov3_multilayer_vitl16"; LOCO=R/"datasets/MVTecLOCO"
HYB=R/"results/auspro_anomaly_maps"; OUTM=R/"results/spro_tuning_maps"; OUT=R/"reports/salad_free"
CATS=["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]
warnings.filterwarnings("ignore"); DEV="cuda"

def pp_maha(Xtr,Xte):
    """per-position Maha: train per-patch cov → test per-patch dist. (GPU). returns (Nte,P)."""
    Xtr=np.nan_to_num(Xtr,nan=0).astype("f4"); Xte=np.nan_to_num(Xte,nan=0).astype("f4")
    n,P,D=Xtr.shape; t=torch.tensor(Xtr,device=DEV); mu=t.mean(0)  # (P,D)
    cen=t-mu; cov=torch.einsum("npi,npj->pij",cen,cen)/max(n-1,1)
    cov+=1e-3*torch.eye(D,device=DEV).unsqueeze(0); prec=torch.linalg.pinv(cov)
    te=torch.tensor(Xte,device=DEV)-mu; sc=torch.einsum("npd,pde,npe->np",te,prec,te)
    return sc.cpu().numpy()

def pc_map(Xtr,Xte,bank=40000):
    """global patch-NN min-dist per patch (GPU). returns (Nte,P)."""
    n,P,D=Xtr.shape; fl=Xtr.reshape(-1,D); rng=np.random.default_rng(0)
    gb=torch.tensor(fl[rng.choice(len(fl),min(bank,len(fl)),replace=False)],device=DEV)
    out=np.zeros((len(Xte),P))
    for i in range(len(Xte)):
        q=torch.tensor(Xte[i],device=DEV); out[i]=torch.cdist(q,gb).min(1).values.cpu().numpy()
    return out

def collect_test(cat):
    items=[]
    for sp in ["good","logical_anomalies","structural_anomalies"]:
        for f in sorted((LOCO/cat/"test"/sp).glob("*.png")): items.append((sp,f.stem,f))
    return items

def zmap(m):  # per-image z-norm
    return (m-m.mean())/(m.std()+1e-9)

def save_maps(cat, maps_2d, items, Hgt, Wgt, cfgname):
    vdir=OUTM/cfgname/cat/"test"
    for (sp,idx,_),mp in zip(items,maps_2d):
        (vdir/sp).mkdir(parents=True,exist_ok=True)
        up=cv2.resize(mp,(Wgt,Hgt),interpolation=cv2.INTER_CUBIC)
        mn,mx=up.min(),up.max(); n=((up-mn)/(mx-mn)*65535).astype(np.uint16) if mx>mn else np.zeros_like(up,np.uint16)
        cv2.imwrite(str(vdir/sp/f"{idx}.tiff"),n)

def load_hybrid(cat,items,G):
    """기존 Hybrid 맵을 G×G로 down (융합용)."""
    out=[]
    for sp,idx,_ in items:
        p=HYB/cat/"test"/sp/f"{idx}.tiff"
        if p.exists():
            m=cv2.imread(str(p),cv2.IMREAD_UNCHANGED).astype("f4"); out.append(cv2.resize(m,(G,G)))
        else: out.append(np.zeros((G,G),"f4"))
    return np.array(out)

def main():
    from src.spro_evaluation import evaluate_object
    configs={c:{} for c in ["L17","multiL","multiL_PC","multiL_PC_hyb"]}
    for cat in CATS:
        z=np.load(CACHE/f"{cat}.npz")
        items=collect_test(cat); g0=glob.glob(str(LOCO/cat/"ground_truth/logical_anomalies/*/*.png"))[0]
        Hgt,Wgt=np.array(Image.open(g0)).shape[:2]
        P=z["test_L17"].shape[1]; G=int(math.isqrt(P))
        # per-position Maha per layer
        ppm={L:pp_maha(z[f"train_L{L}"],z[f"test_L{L}"]) for L in [8,17,23]}  # (N,P)
        pc=pc_map(z["train_L17"],z["test_L17"])  # (N,P)
        hyb=load_hybrid(cat,items,G).reshape(len(items),-1)  # (N,P)
        N=len(items)
        def to2d(arr): return [zmap(arr[i].reshape(G,G)) for i in range(N)]
        variants={
            "L17":        ppm[17],
            "multiL":     (zmap2(ppm[8])+zmap2(ppm[17])+zmap2(ppm[23])),
            "multiL_PC":  (zmap2(ppm[8])+zmap2(ppm[17])+zmap2(ppm[23])+zmap2(pc)),
            "multiL_PC_hyb":(zmap2(ppm[8])+zmap2(ppm[17])+zmap2(ppm[23])+zmap2(pc)+zmap2(hyb)),
        }
        for cn,arr in variants.items():
            save_maps(cat,[arr[i].reshape(G,G) for i in range(N)],items,Hgt,Wgt,cn)
        print(f"[maps] {cat} (G={G}, GT {Hgt}x{Wgt})",flush=True)
    # eval
    res={}
    for cn in ["L17","multiL","multiL_PC","multiL_PC_hyb"]:
        per={}
        for cat in CATS:
            try: per[cat]=round(float(evaluate_object(dataset_base_dir=str(LOCO),anomaly_maps_dir=str(OUTM/cn),object_name=cat)["localization"]["auc_spro"]["mean"][0.05]),4)
            except Exception as e: per[cat]=None; print(f"  {cat} err {e}")
        m=float(np.mean([v for v in per.values() if v is not None]))
        res[cn]={"per_cat":per,"mean":round(m,4)}; print(f"== {cn}: 5-cat AU-sPRO {m:.4f} | {per}",flush=True)
    res["baseline_hybrid"]=0.3944
    json.dump(res,open(OUT/"p9_spro_tuning.json","w"),indent=2,ensure_ascii=False)
    print(f"\nbaseline Hybrid 0.3944 → " + " ".join(f"{k}={res[k]['mean']:.4f}" for k in ["L17","multiL","multiL_PC","multiL_PC_hyb"]))
    print(f"[saved] p9_spro_tuning.json")

def zmap2(arr):  # per-image z-norm, (N,P) → (N,P)
    return (arr-arr.mean(1,keepdims=True))/(arr.std(1,keepdims=True)+1e-9)

if __name__=="__main__":
    main()
