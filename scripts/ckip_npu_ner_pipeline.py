import requests, re, json, time, types, os
import numpy as np, torch
from rknnlite.api import RKNNLite
from ckip_transformers.nlp import CkipNerChunker
from transformers.modeling_outputs import TokenClassifierOutput
SEQ=int(os.environ.get("SEQ","256"))
LIMIT=int(os.environ.get("LIMIT","0"))
RKNN="/media/nvme/rk-ckip/rknn/ckip_bert_ner.rknn"
print("load rknnlite...", flush=True)
rkl=RKNNLite(); rkl.load_rknn(RKNN); rkl.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
print("load CkipNerChunker...", flush=True)
ner=CkipNerChunker(model="bert-base", device=-1)  # model=1 => bert-base
def patched(self,input_ids,attention_mask=None,token_type_ids=None,return_dict=True,**kw):
    ii=input_ids; B,L=ii.shape
    am=attention_mask if attention_mask is not None else torch.ones_like(ii)
    tt=token_type_ids if token_type_ids is not None else torch.zeros_like(ii)
    if L>SEQ: ii,am,tt=ii[:,:SEQ],am[:,:SEQ],tt[:,:SEQ]; L=SEQ
    pad=SEQ-L
    if pad>0:
        ii=torch.cat([ii,torch.zeros(B,pad,dtype=torch.long)],1); am=torch.cat([am,torch.zeros(B,pad,dtype=torch.long)],1); tt=torch.cat([tt,torch.zeros(B,pad,dtype=torch.long)],1)
    out=rkl.inference(inputs=[ii.numpy().astype(np.int64),am.numpy().astype(np.int64),tt.numpy().astype(np.int64)])
    lg=torch.from_numpy(out[0])[:,:L,:]
    return TokenClassifierOutput(logits=lg) if return_dict else (lg,)
ner.model.forward=types.MethodType(patched,ner.model)
print("ready (NPU)", flush=True)
TYPEMAP={"PERSON":"人物","ORG":"機構","GPE":"地點","LOC":"地點","FAC":"場館","EVENT":"事件","PRODUCT":"產品","NORP":"族群","WORK_OF_ART":"作品","LAW":"法律","LANGUAGE":"語言"}
def strip(s): return re.sub(r"\s+"," ",re.sub(r"<[^>]+>"," ",s or "")).strip()
BASE="https://beta.wowsight.tw"
out=open("/media/nvme/ckip_npu_entities.jsonl","w")
page=1; total=0; t0=time.time()
while True:
    r=requests.get(BASE+"/api/posts",params={"limit":100,"page":page,"depth":0},timeout=30); d=r.json(); docs=d.get("docs",[])
    if not docs: break
    texts=[(p["id"],(p.get("title") or "")+" "+strip(p.get("content",""))[:150]) for p in docs]
    try:
        res=ner([t[1] for t in texts], batch_size=1)
    except Exception as e:
        print("NER err page",page,e,flush=True); res=[[] for _ in texts]
    for (pid,_),es in zip(texts,res):
        el=[{"type":TYPEMAP.get(e[1],e[1]),"word":e[0]} for e in es if e[1] in TYPEMAP]
        if el: out.write(json.dumps({"post_id":pid,"entities":el},ensure_ascii=False)+"\n")
    total+=len(docs)
    if total%500==0: print("progress",total,"%.0fs"%(time.time()-t0),flush=True)
    if LIMIT and total>=LIMIT: break
    if d.get("totalPages",0) and page>=d["totalPages"]: break
    page+=1
out.close(); print("FINISHED total",total,flush=True)
