#!/usr/bin/env python3.12
"""
LOCO sPRO full-res held-out 재평가 (Phase 18) — 절대 headline leakage-free 확정
============================================================================
p17은 binary-PRO 256 근사(selection 검증). 본 단계는 **full-res LOCO-sPRO(saturation, evaluate_object)**로
val/test split 재평가. subset GT+맵을 symlink로 구성(read_maps가 GT마다 맵 요구).
config [L17hi_PC_hyb(선택), L17_2PC_hyb(runner-up)] × split [val,test].
→ val로 선택 후 test 보고가 full-res에서도 성립하는지, test-half 절대값이 full-test 0.5287과 일치하는지.
"""
import sys, os, shutil, numpy as np, glob, json, warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
sys.path.insert(0,".")
warnings.filterwarnings("ignore")
R=Path("/workspace/ai-vision-research"); LOCO=R/"datasets/MVTecLOCO"; OUT=R/"reports/salad_free"
TMP=R/"results/heldout_fullres"; CATS=["breakfast_box","juice_bottle","pushpins","screw_bag","splicing_connectors"]
# 선택 config(L17hi)만 — compute_auc_spro가 realpath로 good 매칭 → 맵은 copy 필수(symlink 불가)
CONFIGS={"L17hi_PC_hyb":R/"results/spro_hires_maps/L17hi_PC_hyb"}

def split_ids(cat):
    lst=[]
    for typ in ["good","logical_anomalies","structural_anomalies"]:
        for f in sorted((LOCO/cat/"test"/typ).glob("*.png")): lst.append((typ,f.stem))
    types=[t for t,_ in lst]; idx=np.arange(len(lst))
    val,test=train_test_split(idx,test_size=0.5,random_state=42,stratify=types)
    return {"val":[lst[i] for i in sorted(val)],"test":[lst[i] for i in sorted(test)]}

def ln(src,dst):
    dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    os.symlink(src,dst)

def build(cat, ids, split, cfgdir, cfgname):
    gtb=TMP/f"gt_{split}"/cat; mapb=TMP/f"map_{split}_{cfgname}"/cat
    # defects_config
    ln(LOCO/cat/"defects_config.json", gtb/"defects_config.json")
    for typ,idn in ids:
        # map COPY (realpath가 subset 내부여야 compute_auc_spro good 매칭 성공)
        dst=mapb/"test"/typ/f"{idn}.tiff"; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): shutil.copy2(cfgdir/cat/"test"/typ/f"{idn}.tiff", dst)
        # GT symlink (logical/structural only; good은 GT 없음; read_from_png_dir은 glob이라 symlink OK)
        if typ!="good":
            ln(LOCO/cat/"ground_truth"/typ/idn, gtb/"ground_truth"/typ/idn)
    return str(TMP/f"gt_{split}"), str(TMP/f"map_{split}_{cfgname}")

def main():
    from src.spro_evaluation import evaluate_object
    res={}
    for cfgname,cfgdir in CONFIGS.items():
        for split in ["val","test"]:
            per={}
            for cat in CATS:
                ids=split_ids(cat)[split]
                gtbase,mapbase=build(cat,ids,split,cfgdir,cfgname)
                try:
                    per[cat]=round(float(evaluate_object(dataset_base_dir=gtbase,anomaly_maps_dir=mapbase,object_name=cat)["localization"]["auc_spro"]["mean"][0.05]),4)
                except Exception as e:
                    per[cat]=None; print(f"  {cfgname}/{split}/{cat} err {e}")
            m=float(np.mean([v for v in per.values() if v is not None]))
            res[f"{cfgname}_{split}"]={"mean":round(m,4),"per_cat":per}
            print(f"  {cfgname:14s} {split:4s} full-res sPRO = {m:.4f}",flush=True)
    # held-out 판정
    val_sel=max(CONFIGS,key=lambda c:res[f"{c}_val"]["mean"])
    res["_heldout"]={"selected_by_val":val_sel,"val":res[f"{val_sel}_val"]["mean"],
                     "heldout_test":res[f"{val_sel}_test"]["mean"],"full_test_ref_L17hi":0.5287}
    json.dump(res,open(OUT/"p18_heldout_fullres.json","w"),indent=2)
    print(f"\n=== full-res held-out: val-선택 = '{val_sel}' → held-out test sPRO = {res[f'{val_sel}_test']['mean']:.4f}")
    print(f"    (full-test L17hi 0.5287 참조; test-half가 근접하면 절대 headline도 leakage-free)")
    print("[saved] p18_heldout_fullres.json")

if __name__=="__main__":
    main()
