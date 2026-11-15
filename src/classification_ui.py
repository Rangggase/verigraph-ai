"""Read-only view/inference for a trusted locally trained claim-only baseline."""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st

from src.errors import friendly_error
from src.model_integrity import ModelIntegrityError, load_trusted_model

ROOT=Path(__file__).resolve().parents[1]

def render_classification_tab():
    st.subheader('v0.5 — Claim-only ML/DL baselines')
    st.warning('Claim-only models DO NOT check factual evidence and cannot be treated as reliable fact-checkers. No KG features are used here.')
    st.caption('Train from the terminal first, then inspect evaluation here. Pilot and subset runs are NOT official benchmarks.')
    base=ROOT/'outputs'/'ml_v05'
    folders=[('English / AVeriTeC pilot',base/'averitec_pilot'),
             ('English / AVeriTeC official',base/'averitec'),
             ('Chinese / CFEVER',base/'cfever')]
    available=[(name,folder) for name,folder in folders if (folder/'metrics.json').is_file() and (folder/'model.joblib').is_file()]
    if not available:
        st.info('No trained model found. See docs/UPGRADE_V05.md for command-line training. The included CFEVER train/dev files are ready to use.')
        return
    choice=st.selectbox('Trained baseline',list(range(len(available))),format_func=lambda i:available[i][0])
    name,folder=available[choice]
    try:
        report=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
    except (OSError,ValueError) as exc:
        st.error(friendly_error(exc,'Pembacaan laporan'));return
    c1,c2,c3=st.columns(3)
    c1.metric('Dev Macro-F1',f"{report['metrics']['macro_f1']:.3f}")
    c2.metric('Majority Macro-F1',f"{report['majority_baseline']['macro_f1']:.3f}")
    c3.metric('Dev samples',report['data']['dev_size'])
    if report['data']['mode'] != 'OFFICIAL_SPLIT' or report['data']['limited']:
        st.warning('Exploratory/pilot/subset run — not an independently validated full benchmark.')
    st.write('Labels:', ', '.join(report['labels']))
    st.json({'mode':report['data']['mode'],'split_checks':{'overlap':report['data']['normalized_claim_overlap']},
             'distribution':{'train':report['data']['train_labels'],'dev':report['data']['dev_labels']},
             'confusion_matrix':report['metrics']['confusion_matrix']})
    st.caption('Classification report and all settings are in outputs/ml_v05/<dataset>/metrics.json')
    user_claim=st.text_area('Input a new claim (exploratory model prediction only)',
           value=('北京是中国的首都。' if folder.name=='cfever' else 'Hunter Biden had no experience in the energy sector.'),key=f'v05_claim_{choice}',height=90)
    if st.button('Predict claim-text label (NOT fact verification)',key=f'v05_predict_{choice}'):
        if not user_claim.strip():
            st.warning('Enter a claim.');return
        try:
            # Only load project-local, signature-verified artifacts created by our CLI.
            model=load_trusted_model(folder/'model.joblib')
            label=str(model.predict([user_claim])[0])
            st.info(f'Claim-only model prediction: **{label}**. This is NOT a verified verdict.')
            probs=model.predict_proba([user_claim])[0]
            st.dataframe([{'label':str(cls),'score':round(float(p),4)} for cls,p in zip(model.classes_,probs)],
                          hide_index=True,use_container_width=True)
        except ModelIntegrityError as exc:
            st.error(f'Model ditolak: {exc} Latih ulang dengan scripts/train_ml_baseline.py.')
        except Exception as exc:
            st.error(friendly_error(exc, 'Prediksi'))
