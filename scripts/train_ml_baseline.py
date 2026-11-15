"""Train/evaluate claim-only TF-IDF+LogisticRegression, with explicit split guards."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from src.classification_data import LABELS, prepare_split


def build_vectorizer(dataset):
    if dataset == 'cfever':
        return TfidfVectorizer(analyzer='char', ngram_range=(2,4), min_df=2,
                               max_features=100_000, sublinear_tf=True)
    return TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2,
                           max_features=60_000, sublinear_tf=True, strip_accents='unicode')


def train(args):
    train_rows, dev_rows, data_report = prepare_split(
        args.dataset, args.train, args.dev, args.pilot, args.seed,
        args.max_train, args.max_dev)
    xtr = [r['claim'] for r in train_rows]
    ytr = [r['label'] for r in train_rows]
    xdev = [r['claim'] for r in dev_rows]
    ydev = [r['label'] for r in dev_rows]
    labels = LABELS[args.dataset]
    vectorizer = build_vectorizer(args.dataset)
    clf = make_pipeline(vectorizer, LogisticRegression(max_iter=1000,
          class_weight='balanced', random_state=args.seed))
    clf.fit(xtr, ytr)
    pred = clf.predict(xdev)
    majority = DummyClassifier(strategy='most_frequent').fit(np.asarray(xtr).reshape(-1,1), ytr)
    base_pred = majority.predict(np.asarray(xdev).reshape(-1,1))
    report = {
        'task': 'claim_only_text_classification_NOT_fact_verification',
        'data': data_report,
        'feature_columns': ['claim'],
        'excluded_columns': ['label','evidence','questions','justification','evidence_json','fact_check_url'],
        'labels': labels,
        'model': {'name': 'tfidf_logistic_regression', 'dataset': args.dataset,
                  'tfidf_analyzer': vectorizer.analyzer, 'vocabulary_size': len(clf.named_steps['tfidfvectorizer'].vocabulary_)},
        'metrics': {'macro_f1': float(f1_score(ydev, pred, labels=labels, average='macro', zero_division=0)),
                    'accuracy': float(accuracy_score(ydev, pred)),
                    'confusion_matrix': confusion_matrix(ydev, pred, labels=labels).tolist(),
                    'classification_report': classification_report(ydev, pred, labels=labels, output_dict=True, zero_division=0)},
        'majority_baseline': {'macro_f1': float(f1_score(ydev, base_pred, labels=labels, average='macro', zero_division=0)),
                              'accuracy': float(accuracy_score(ydev, base_pred))},
        'interpretation': 'Model sees CLAIM TEXT ONLY. No document retrieval, evidence checking, or Knowledge Graph. Predictions are not factual verdicts.',
        'warnings': (['PILOT split: do not advertise score as an independent benchmark'] if args.pilot else []) +
                    (['SUBSET RUN: dev result not an official full-split benchmark'] if data_report['limited'] else []) +
                    (['Training examples deduplicated/conflicting labels removed. Scores not identical to official published baseline protocols.'] if not args.pilot else []),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / 'model.joblib')
    from src.model_integrity import write_sidecar
    write_sidecar(out / 'model.joblib')
    with (out / 'metrics.json').open('w',encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with (out / 'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        import csv
        w=csv.writer(f); w.writerow(['source_id','claim','gold_label','predicted_label'])
        for r,p in zip(dev_rows,pred): w.writerow([r['source_id'],r['claim'],r['label'],p])
    print(f"{args.dataset} mode={data_report['mode']}, train={len(train_rows)}, dev={len(dev_rows)}")
    print(f"Macro-F1: {report['metrics']['macro_f1']:.4f}; majority: {report['majority_baseline']['macro_f1']:.4f}")
    print('Saved',out.resolve())
    return report


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=LABELS,required=True)
    p.add_argument('--train',required=True)
    p.add_argument('--dev')
    p.add_argument('--pilot', action='store_true',help='Exploratory split within train; NOT benchmark')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-train',type=int)
    p.add_argument('--max-dev',type=int)
    p.add_argument('--output',required=True)
    return p

if __name__=='__main__': train(parser().parse_args())
