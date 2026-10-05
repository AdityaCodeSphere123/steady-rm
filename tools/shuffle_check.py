import random, sys, json
from submission import feedback as fb
from tools.track_a_sweep import load
fb.set_params(**json.loads(sys.argv[1]))
tot=0
for s in "scifact nfcorpus fiqa vaswani antique touche2020 scidocs cacm cranfield arguana full".split():
    d=load(s); fb.prepare(d[0]); _,qs,qr,c=d
    mv=0
    for q,t in qs:
        pool=c[q]; seed=[x for x,_ in fb.score_candidates(t,pool,10)]
        ref=[x for x,_ in fb.relevance_model_feedback(t,seed,pool,10)]
        for r in range(4):
            p=list(pool); random.Random(r).shuffle(p) if r<3 else p.reverse()
            sd=list(seed); random.Random(r+9).shuffle(sd)
            mv+=[x for x,_ in fb.relevance_model_feedback(t,sd,p,10)]!=ref
    tot+=mv
print("total B moves under shuffle:", tot)
