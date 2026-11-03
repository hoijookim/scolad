#!/usr/bin/env python3.12
"""
LOCO sPRO 고해상도 (Phase 15) — DINOv3-L res 672 (42×42) per-position Maha + PC
==============================================================================
현 best L17_PC_hyb 0.4953는 DINOv3 21×21(res336). 고해상도(42×42, res672, 4× patch)로 finer map.
거대 캐시 회피: per-cat 추출(in-memory)→Maha+PC 맵→tiff 저장→features free. config: L17_hi+PC_hi+Hybrid.
eval: src/spro_evaluation @0.05. 비교: L17_PC_hyb 0.4953.
"""
import os, sys, numpy as np, glob, math, json, warnings, cv2, torch
os.environ["HF_HUB_OFFLINE"]="1"
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
sys.path.insert(0,"scripts/salad_free"); sys.path.insert(0,".")
import p9_spro_tuning as P9
warnings.filterwarnings("ignore")
R=Path("/workspace/ai-vision-research"); LOCO=R/"datasets/MVTecLOCO"; HYB=R/"results/auspro_anomaly_maps"
MASKROOT=R/"external/ROMAD_baselines/PSAD/data/unet_seed42"  # 미사용; LOCO 이미지 직접
OUTM=R/"results/spro_hires_maps"; OUT=R/"reports/salad_free"
CATS=["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]
MODEL="facebook/dinov3-vitl16-pretrain-lvd1689m"; RES=512; LAYER=17; DEV="cuda"  # 32×32 (현 21×21 2.3×), GPU 메모리 안전
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def collect(cat):
    tr=sorted((LOCO/cat/"train/good").glob("*.png"))+sorted((LOCO/cat/"validation/good").glob("*.png"))
    te=[]
    for sp in ["good","logical_anomalies","structural_anomalies"]:
        for f in sorted((LOCO/cat/"test"/sp).glob("*.png")): te.append((sp,f.stem,f))
    return tr,te

def main():
    print("loading DINOv3-L (HF)...",flush=True)
    model=AutoModel.from_pretrained(MODEL,torch_dtype=torch.float32).to(DEV).eval()
    layers=model.layer if hasattr(model,"layer") else model.encoder.layer
    out={}
    def hook(m,i,o): out['x']=o[0] if isinstance(o,tuple) else o
    layers[LAYER].register_forward_hook(hook)
    tf=transforms.Compose([transforms.Resize((RES,RES),interpolation=transforms.InterpolationMode.BICUBIC),
                           transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def feats(items):
        F=[]
        for it in items:
            path=it if isinstance(it,Path) else it[2]
            t=tf(Image.open(path).convert("RGB")).unsqueeze(0).to(DEV)
            with torch.no_grad(): out.clear(); _=model(t)
            F.append(out['x'][0,5:,:].cpu().numpy().astype("f4"))
        return np.stack(F)
    for cat in CATS:
        tr,te=collect(cat); print(f"=== {cat} === train={len(tr)} test={len(te)}",flush=True)
        Xtr=feats(tr); Xte=feats([t for t in te]); P=Xtr.shape[1]; G=int(math.isqrt(P))
        m17=P9.zmap2(P9.pp_maha(Xtr,Xte)); pc=P9.zmap2(P9.pc_map(Xtr,Xte))
        g0=glob.glob(str(LOCO/cat/"ground_truth/logical_anomalies/*/*.png"))[0]; Hgt,Wgt=np.array(Image.open(g0)).shape[:2]
        # Hybrid 맵 로드(24×24)→G로
        hyb=P9.zmap2(P9.load_hybrid(cat,te,G).reshape(len(te),-1))
        fused=m17+pc+hyb
        items=[(sp,idx,None) for sp,idx,_ in te]
        P9.OUTM=OUTM
        P9.save_maps(cat,[fused[i].reshape(G,G) for i in range(len(te))],items,Hgt,Wgt,"L17hi_PC_hyb")
        print(f"  [maps] {cat} G={G}",flush=True); del Xtr,Xte; torch.cuda.empty_cache()
    from src.spro_evaluation import evaluate_object
    per={}
    for cat in CATS:
        try: per[cat]=round(float(evaluate_object(dataset_base_dir=str(LOCO),anomaly_maps_dir=str(OUTM/"L17hi_PC_hyb"),object_name=cat)["localization"]["auc_spro"]["mean"][0.05]),4)
        except Exception as e: per[cat]=None; print(f"  {cat} err {e}")
    m=float(np.mean([v for v in per.values() if v is not None]))
    res={"L17hi_PC_hyb":{"per_cat":per,"mean":round(m,4)},"ref":{"L17_PC_hyb_336":0.4953,"hybrid":0.3944}}
    json.dump(res,open(OUT/"p15_loco_hires.json","w"),indent=2)
    print(f"\n== 고해상도 res{RES}(42×42) L17hi_PC_hyb: 5-cat {m:.4f} | {per}  (vs res336 L17_PC_hyb 0.4953)")
    print("[saved] p15_loco_hires.json")

if __name__=="__main__":
    main()
