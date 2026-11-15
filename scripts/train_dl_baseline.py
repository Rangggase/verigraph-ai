"""Optional lightweight multilingual Transformer fine-tuning (claim-only).

Run from project root, AFTER ML baseline. Downloads model weights first time.
Requires torch + transformers. CPU users should use --max-train for a quick smoke test.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.classification_data import LABELS,prepare_split


def train(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    train_rows, dev_rows, summary = prepare_split(args.dataset,args.train,args.dev,
                args.pilot,args.seed,args.max_train,args.max_dev)
    labels=LABELS[args.dataset]
    label2id={name:i for i,name in enumerate(labels)}
    tokenizer=AutoTokenizer.from_pretrained(args.checkpoint)
    model=AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,num_labels=len(labels),id2label={i:name for i,name in enumerate(labels)},
        label2id=label2id, ignore_mismatched_sizes=True)

    class ClaimDataset(Dataset):
        def __init__(self,rows): self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i]
            enc=tokenizer(row['claim'],truncation=True,max_length=args.max_length,
                          padding='max_length',return_tensors='pt')
            return {k:v.squeeze(0) for k,v in enc.items()} | {'labels':torch.tensor(label2id[row['label']],dtype=torch.long)}

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader=DataLoader(ClaimDataset(train_rows),batch_size=args.batch_size,shuffle=True)
    dev_loader=DataLoader(ClaimDataset(dev_rows),batch_size=args.batch_size,shuffle=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch={k:v.to(device) for k,v in batch.items()}
            result=model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            losses.append(result.loss.item())
        model.eval(); truth=[];pred=[]
        with torch.no_grad():
            for batch in dev_loader:
                batch={k:v.to(device) for k,v in batch.items()}
                logits=model(**batch).logits
                truth.extend(batch['labels'].cpu().tolist())
                pred.extend(logits.argmax(dim=1).cpu().tolist())
        f1=float(f1_score(truth,pred,labels=list(range(len(labels))),average='macro',zero_division=0))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'dev_macro_f1':f1})
        print(f'Epoch {epoch+1} loss={history[-1]["train_loss"]:.4f} macro-F1={f1:.4f}',flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(out/'model');tokenizer.save_pretrained(out/'model')
    metrics={'task':'claim_only_text_classification_NOT_fact_verification',
             'data':summary,'labels':labels,'model':{'name':'transformer_finetune','checkpoint':args.checkpoint,'device':str(device)},
             'feature_columns':['claim'], 'history':history,
             'metrics':{'macro_f1':history[-1]['dev_macro_f1'],
                'accuracy':float(accuracy_score(truth,pred)),
                'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(len(labels)))).tolist(),
                'classification_report':classification_report(truth,pred,labels=list(range(len(labels))),
                                              target_names=labels,output_dict=True,zero_division=0)},
             'warnings':(['PILOT: not official benchmark'] if args.pilot else []) +
                 (['SUBSET: not official full-split benchmark'] if summary['limited'] else []),
             'interpretation':'No evidence retrieval, KG, or verdict verification; text-only predictive baseline.'}
    with (out/'metrics.json').open('w',encoding='utf8') as f:json.dump(metrics,f,ensure_ascii=False,indent=2)
    print('Saved:',out)
    return metrics


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=LABELS,required=True)
    p.add_argument('--train',required=True);p.add_argument('--dev');p.add_argument('--pilot',action='store_true')
    p.add_argument('--output',required=True);p.add_argument('--checkpoint',default='distilbert-base-multilingual-cased')
    p.add_argument('--epochs',type=int,default=2);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-length',type=int,default=160);p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--max-train',type=int);p.add_argument('--max-dev',type=int);p.add_argument('--seed',type=int,default=42)
    return p

if __name__=='__main__':train(parser().parse_args())
