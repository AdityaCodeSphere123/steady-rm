import sys, json
from submission import feedback as fb
from tools.track_a_sweep import load, evaluate
SETS=sys.argv[2].split(",") if len(sys.argv)>2 else "scifact nfcorpus fiqa vaswani antique touche2020 scidocs cacm cranfield arguana full".split()
configs=json.loads(open(sys.argv[1]).read())
base=dict({"ax_z": 1}, **fb.PARAMS); base.update(json.loads(sys.argv[3]) if len(sys.argv)>3 else {})
for c in configs.values():
    for key in c:
        base.setdefault(key, fb.get_param(key))
res={k:{} for k in configs}
for s in SETS:
    d=load(s); fb.set_params(**base); fb.prepare(d[0])
    for name,c in configs.items():
        fb.set_params(**dict(base,**c)); r=evaluate(d, with_fb=True); res[name][s]=r
names=list(configs); ref=names[0]
print("set        "+" ".join(f"{n:>26s}" for n in names))
for s in SETS:
    print(f"{s:10s} "+" ".join(f"{res[n][s][0]:.4f}/{res[n][s][1]:.4f}/{res[n][s][2]:.4f}".rjust(26) for n in names))
for n in names:
    dn=[res[n][s][0]-res[ref][s][0] for s in SETS]; dm=[res[n][s][1]-res[ref][s][1] for s in SETS]; db=[res[n][s][2]-res[ref][s][2] for s in SETS]
    print(f"{n:20s} " + " ".join(f"{x:+.4f}" for x in db) + f" dA {sum(dn)/len(dn):+.5f} dMAP {sum(dm)/len(dm):+.5f} dB {sum(db)/len(db):+.5f} worstA {min(dn):+.4f} sets>=0: {sum(x>=-1e-9 for x in dn)}/{len(dn)}")
