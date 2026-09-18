#!/usr/bin/env python3.12
"""Pixel-level AUROC (P-AUROC) — localization map의 픽셀 vs GT. Ours vs Hybrid baseline."""
import numpy as np, glob, cv2, json, warnings
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R=Path("/workspace/ai-vision-research"); LOCO=R/"datasets/MVTecLOCO"; OUT=R/"reports/salad_free"; RES=256
CATS=["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]
CONFIGS={"Ours (L17hi_PC_hyb)":R/"results/spro_hires_maps/L17hi_PC_hyb",
         "Hybrid (DINOv2-B Maha)":R/"results/auspro_anomaly_maps"}
SP={"good":"good","logical":"logical_anomalies","structural":"structural_anomalies"}

def gt_bin(cat,typ,idx):
    if typ=="good": return np.zeros((RES,RES),np.uint8)
    gg=sorted(glob.glob(str(LOCO/cat/"ground_truth"/SP[typ]/idx/"*.png")))
    if not gg: return np.zeros((RES,RES),np.uint8)
    acc=None
    for f in gg:
        m=(np.array(Image.open(f))>0).astype(np.uint8); acc=m if acc is None else np.maximum(acc,m)
    return (cv2.resize(acc,(RES,RES),interpolation=cv2.INTER_NEAREST)>0).astype(np.uint8)

def load_map(cdir,cat,typ,idx):
    p=cdir/cat/"test"/SP[typ]/f"{idx}.tiff"
    if not p.exists(): return None
    return cv2.resize(cv2.imread(str(p),cv2.IMREAD_UNCHANGED).astype("f4"),(RES,RES))

def main():
    res={}
    for cn,cdir in CONFIGS.items():
        per={};
        for cat in CATS:
            ids={t:[f.stem for f in sorted((LOCO/cat/"test"/SP[t]).glob("*.png"))] for t in SP}
            # 픽셀 모으기
            scores={"all":[], "logical":[], "structural":[]}; labels={"all":[],"logical":[],"structural":[]}
            good_s=[]; good_l=[]
            for t in ["good","logical","structural"]:
                for idx in ids[t]:
                    mp=load_map(cdir,cat,t,idx)
                    if mp is None: continue
                    g=gt_bin(cat,t,idx).ravel(); s=mp.ravel()
                    if t=="good": good_s.append(s); good_l.append(g)
                    else:
                        scores[t].append(s); labels[t].append(g)
                    scores["all"].append(s); labels["all"].append(g)
            gs=np.concatenate(good_s); gl=np.concatenate(good_l)
            o={}
            for kind in ["all","logical","structural"]:
                if kind=="all":
                    S=np.concatenate(scores["all"]); L=np.concatenate(labels["all"])
                else:
                    S=np.concatenate([gs]+scores[kind]); L=np.concatenate([gl]+labels[kind])
                o[kind]=round(float(roc_auc_score(L,S)),4) if L.max()>0 else None
            per[cat]=o
            print(f"  [{cn}] {cat:22s} all={o['all']} log={o['logical']} str={o['structural']}",flush=True)
        m_all=float(np.mean([per[c]["all"] for c in CATS]))
        m_log=float(np.mean([per[c]["logical"] for c in CATS]))
        m_str=float(np.mean([per[c]["structural"] for c in CATS]))
        res[cn]={"per_cat":per,"mean_all":round(m_all,4),"mean_logical":round(m_log,4),"mean_structural":round(m_str,4)}
        print(f"== {cn}: P-AUROC all={m_all:.4f} log={m_log:.4f} str={m_str:.4f}",flush=True)
    json.dump(res,open(OUT/"p21_pixel_auroc.json","w"),indent=2,ensure_ascii=False)
    print(f"[saved] {OUT/'p21_pixel_auroc.json'}")

if __name__=="__main__":
    main()
