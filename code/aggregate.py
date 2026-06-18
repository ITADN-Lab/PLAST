#!/usr/bin/env python3
import json, glob, sys, math, os
from statistics import mean, pstdev

def load(resdir, methods, key="late_acc"):
    out={}
    for m in methods:
        vals=[]
        for f in sorted(glob.glob(f"{resdir}/{m}_s*.json")):
            try: vals.append(json.load(open(f))[key])
            except: pass
        if vals: out[m]=vals
    return out

def welch_t(a,b):
    if len(a)<2 or len(b)<2: return float("nan")
    ma,mb=mean(a),mean(b)
    va=pstdev(a)**2*len(a)/(len(a)-1); vb=pstdev(b)**2*len(b)/(len(b)-1)
    se=math.sqrt(va/len(a)+vb/len(b))
    return (ma-mb)/se if se>0 else float("nan")

resdir=sys.argv[1] if len(sys.argv)>1 else "/home/hadoop/workstation/md/AAAI/results/p0_gate"
methods=sys.argv[2].split(",") if len(sys.argv)>2 else ["adam","muon","adamo","ptc","ptc_muon","ptc_adamo"]
d=load(resdir,methods)
print(f"=== {os.path.basename(resdir)}  late_acc (mean±std, n) ===")
rows=sorted(d.items(), key=lambda kv:-mean(kv[1]))
for m,v in rows:
    print(f"  {m:11s} {mean(v):6.2f} ± {pstdev(v):4.2f}  (n={len(v)})  seeds={[round(x,1) for x in v]}")
# 关键 t 检验
for a,b in [("ptc_muon","adamo"),("ptc_muon","muon"),("ptc","adamo"),("ptc_adamo","adamo")]:
    if a in d and b in d:
        t=welch_t(d[a],d[b]); dm=mean(d[a])-mean(d[b])
        print(f"  Δ {a} vs {b}: {dm:+.2f}  t={t:.2f}  {'***' if abs(t)>3 else '*' if abs(t)>2 else 'ns'}")
