"""VeriGraph AI v0.5.2 — safer gradient-overflow handling and FP32 full training.

This is an experimental claim-only classifier, NOT evidence-based fact verification.
Run from the project root. Requires src/classification_data.py from v0.5.

Modes:
  diagnostic: deliberately overfit a tiny balanced training sample; evaluates SAME
              examples to check whether gradients and parameter updates work.
  smoke: train on small balanced train/dev subsets to check end-to-end execution.
  full: train on entire cleaned source train, evaluate official source dev.

Outputs use distinct directories. A class-weighted objective is an experimental
intervention, not proof that model collapse is fixed.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classification_data import LABELS, prepare_split


def stratified_subset(rows: list[dict], size: int | None, seed: int) -> list[dict]:
    """Balanced deterministic subset for diagnostics ONLY (not final validation)."""
    if size is None or size >= len(rows):
        return rows
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        by_label[r['label']].append(r)
    labels = sorted(by_label)
    if size < len(labels):
        raise ValueError('Subset must include at least one item per class')
    for lab in labels:
        rng.shuffle(by_label[lab])
    result = []
    while len(result) < size:
        progress = False
        for lab in labels:
            if by_label[lab] and len(result) < size:
                result.append(by_label[lab].pop())
                progress = True
        if not progress:
            break
    rng.shuffle(result)
    return result


def normalized_sqrt_inverse_weights(train_y: list[int], n_labels: int):
    import torch
    counts = Counter(train_y)
    vals = [1. / math.sqrt(max(1, counts[i])) for i in range(n_labels)]
    mean = sum(vals) / len(vals)
    return torch.tensor([v / mean for v in vals], dtype=torch.float32)


def summarize_predictions(ids, labels):
    counts = Counter(int(i) for i in ids)
    return {lab: int(counts.get(i, 0)) for i, lab in enumerate(labels)}


def safe_float(x):
    return float(x.detach().cpu().item() if hasattr(x, 'detach') else x)


def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = LABELS[args.dataset]
    label2id = {label: i for i, label in enumerate(labels)}

    train_rows, dev_rows, base_summary = prepare_split(args.dataset, args.train, args.dev,
                                                        pilot=False, seed=args.seed)
    if args.mode == 'diagnostic':
        train_rows = stratified_subset(train_rows, args.max_train or 24, args.seed)
        dev_rows = list(train_rows)  # INTENTIONAL train-as-eval memorization test, NOT benchmark.
    elif args.mode == 'smoke':
        train_rows = stratified_subset(train_rows, args.max_train or 600, args.seed)
        dev_rows = stratified_subset(dev_rows, args.max_dev or 300, args.seed + 1)
    elif args.mode != 'full':
        raise ValueError(args.mode)

    print('MODE:', args.mode, 'device:', device, flush=True)
    print('Train:', len(train_rows), 'Dev:', len(dev_rows), flush=True)
    print('Train labels:', dict(Counter(r['label'] for r in train_rows)), flush=True)
    print('Dev labels:', dict(Counter(r['label'] for r in dev_rows)), flush=True)
    print('Note: A diagnostic evaluates its training examples deliberately; it is NOT a generalization score.', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    sample = [r['claim'] for r in train_rows[:min(6, len(train_rows))]]
    for i, t in enumerate(sample[:3]):
        ids = tokenizer(t, truncation=True, max_length=args.max_length)['input_ids']
        print(f'Tokenization sample {i}: {len(ids)} tokens / {len(t)} chars / {sum(tok == tokenizer.unk_token_id for tok in ids)} UNK',flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=len(labels), id2label={i:l for i,l in enumerate(labels)},
        label2id=label2id,
    ).to(device)
    # Do not freeze encoder. Check that the encoder and classifier are trainable.
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Parameters trainable/total:', n_train, '/', n_all, flush=True)
    if n_train < n_all * 0.9:
        raise RuntimeError('Most transformer parameters frozen unexpectedly')

    class EncodedDataset(Dataset):
        def __init__(self, rows):
            self.items = tokenizer([r['claim'] for r in rows], truncation=True,
                                   max_length=args.max_length, padding=False)
            self.labels = [label2id[r['label']] for r in rows]
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return {k:self.items[k][idx] for k in self.items} | {'labels':self.labels[idx]}

    def collate(batch):
        labs = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
        enc = [{k:v for k,v in b.items() if k != 'labels'} for b in batch]
        result = tokenizer.pad(enc, padding=True, return_tensors='pt')
        result['labels'] = labs
        return result

    train_loader = DataLoader(EncodedDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(EncodedDataset(dev_rows), batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    train_y = [label2id[r['label']] for r in train_rows]
    weights = (normalized_sqrt_inverse_weights(train_y, len(labels))
               if args.class_weighting == 'sqrt_inverse'
               else torch.ones(len(labels), dtype=torch.float32)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    print('Class loss weights:', {labels[i]: round(float(v), 4) for i,v in enumerate(weights)},flush=True)
    print('Optimizer steps:', total_steps, 'warmup:', warmup_steps,flush=True)
    print('AMP:', bool(args.amp and device.type=='cuda'),flush=True)

    use_amp = bool(args.amp and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    best_score = -1
    best_epoch = 0
    history = []
    best_metrics = None
    patience_left = args.patience
    first_update_checked = False
    update_steps = 0
    skipped_amp_overflows = 0
    consecutive_amp_overflows = 0

    def evaluate(rows_loader):
        model.eval()
        truth, pred = [], []
        with torch.no_grad():
            for data in rows_loader:
                data = {k:v.to(device) for k,v in data.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(**{k:v for k,v in data.items() if k!='labels'}).logits
                pred.extend(logits.argmax(-1).cpu().tolist())
                truth.extend(data['labels'].cpu().tolist())
        return {
            'macro_f1': float(f1_score(truth,pred,average='macro',labels=list(range(len(labels))),zero_division=0)),
            'accuracy': float(accuracy_score(truth,pred)),
            'confusion_matrix': confusion_matrix(truth,pred,labels=list(range(len(labels)))).tolist(),
            'classification_report': classification_report(truth,pred,labels=list(range(len(labels))),
                                                           target_names=labels,output_dict=True,zero_division=0),
            'prediction_counts': summarize_predictions(pred,labels),
        }

    for ep in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        grad_norm_first = None
        for i, batch in enumerate(train_loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**{k:v for k,v in batch.items() if k!='labels'}).logits
                loss = loss_fn(logits, batch['labels'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss epoch={ep+1} step={i+1}')
            scaler.scale(loss/args.grad_accum).backward()
            losses.append(safe_float(loss))
            if ((i+1) % args.grad_accum == 0) or (i+1 == len(train_loader)):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if ep == 0 and grad_norm_first is None:
                    grad_norm_first = safe_float(grad_norm)
                if not torch.isfinite(grad_norm):
                    if not use_amp:
                        raise RuntimeError(
                            'Non-finite gradient norm in FP32; investigate data, loss and learning rate'
                        )
                    # GradScaler.unscale_ recorded inf/nan already. scaler.step()
                    # skips optimizer.step() and scaler.update() reduces the scale.
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    optimizer.zero_grad(set_to_none=True)
                    skipped_amp_overflows += 1
                    consecutive_amp_overflows += 1
                    print(f'AMP_OVERFLOW_SKIP epoch={ep+1} batch={i+1} '
                          f'scale={scale_before:.1f}->{scale_after:.1f}', flush=True)
                    if consecutive_amp_overflows >= 30:
                        raise RuntimeError('Repeated AMP overflows (30 consecutive); rerun without --amp')
                    # Do NOT advance LR scheduler when optimizer step was skipped.
                    continue
                consecutive_amp_overflows = 0
                if not first_update_checked:
                    watch = next((p for name,p in model.named_parameters() if ('classifier' in name and p.requires_grad)), None)
                    if watch is None:
                        watch = next(p for p in model.parameters() if p.requires_grad)
                    before = watch.detach().clone()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_steps += 1
                if not first_update_checked:
                    delta = safe_float((watch.detach()-before).abs().max())
                    print('First update: gradient_norm=',round(safe_float(grad_norm),5),
                          'max_parameter_delta=',round(delta,9),flush=True)
                    if delta <= 0:
                        print('NOTE: first update below parameter float precision; check gradient and loss across subsequent batches.',flush=True)
                    first_update_checked = True
        values = evaluate(dev_loader)
        record = {'epoch':ep+1, 'train_loss':round(float(np.mean(losses)),6),
                  'dev_macro_f1':values['macro_f1'], 'dev_accuracy':values['accuracy'],
                  'prediction_counts':values['prediction_counts'],
                  'first_batch_gradient_norm':grad_norm_first,
                  'skipped_amp_overflows_so_far':skipped_amp_overflows,
                  'learning_rate_end':optimizer.param_groups[0]['lr']}
        history.append(record)
        print('EPOCH',ep+1,'loss',record['train_loss'],'macro_f1',round(record['dev_macro_f1'],4),
              'predictions',record['prediction_counts'],flush=True)
        if values['macro_f1'] > best_score + 1e-5:
            best_score=values['macro_f1']; best_epoch=ep+1; best_metrics=values
            model.save_pretrained(out/'model')
            tokenizer.save_pretrained(out/'model')
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print('Early stop after no improvement',flush=True)
                break

    result = {
        'task': 'claim_only_text_classification_NOT_fact_verification',
        'mode':args.mode,
        'valid_benchmark_comparison': args.mode == 'full',
        'data': base_summary,
        'effective_train_size':len(train_rows), 'effective_dev_size':len(dev_rows),
        'labels': labels,
        'model': {'name':'xlm_roberta_class_weighted_experiment', 'checkpoint':args.checkpoint,
                  'device':str(device), 'class_weighting':args.class_weighting},
        'hyperparameters':{'epochs_requested':args.epochs,'batch_size':args.batch_size,
                           'max_length':args.max_length,'lr':args.lr,'warmup_ratio':args.warmup_ratio,
                           'grad_accum':args.grad_accum,'seed':args.seed,'amp':use_amp},
        'history':history, 'best_epoch':best_epoch, 'metrics': best_metrics,
        'optimizer_steps_completed':update_steps,
        'amp_overflow_steps_skipped':skipped_amp_overflows,
        'warnings': (["Diagnostic evaluated TRAIN items intentionally; NOT benchmark"] if args.mode=='diagnostic'
                     else ['Smoke subset, NOT benchmark'] if args.mode=='smoke' else []) +
                    (["Single-class prediction collapse remains; do not present as functioning multiclass model"]
                     if sum(v>0 for v in best_metrics['prediction_counts'].values())==1 else []),
        'interpretation':'Text-only class prediction; no evidence verification or graph reasoning.'
    }
    with (out/'metrics.json').open('w',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print('BEST',best_epoch, 'F1',round(best_score,5),flush=True)
    print('Output:',out,flush=True)


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=LABELS,default='cfever')
    p.add_argument('--train',required=True)
    p.add_argument('--dev',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['diagnostic','smoke','full'],default='diagnostic')
    p.add_argument('--checkpoint',default='xlm-roberta-base')
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--max-length',type=int,default=160)
    p.add_argument('--max-train',type=int)
    p.add_argument('--max-dev',type=int)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--warmup-ratio',type=float,default=0.06)
    p.add_argument('--weight-decay',type=float,default=0.01)
    p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--max-grad-norm',type=float,default=1.0)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--class-weighting',choices=['none','sqrt_inverse'],default='sqrt_inverse')
    p.add_argument('--amp',action='store_true')
    return p


if __name__=='__main__':
    main(parser().parse_args())
