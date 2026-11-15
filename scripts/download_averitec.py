"""Optional official AVeriTeC train/dev download; NEVER use test verdicts."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    raise SystemExit('Install: python -m pip install huggingface_hub') from exc

out=ROOT/'data/datasets/averitec'
out.mkdir(parents=True,exist_ok=True)
for split in ('train','dev'):
    source=hf_hub_download(repo_id='chenxwh/AVeriTeC',repo_type='model',filename=f'data/{split}.json')
    dest=out/f'{split}.json'
    with open(source,encoding='utf-8') as f:
        data=json.load(f)
    if not isinstance(data,list) or not all(isinstance(x,dict) and 'claim' in x and 'label' in x for x in data):
        raise ValueError(f'Unexpected schema in {split}')
    with open(dest,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False)
    print(f'{split}: {len(data)} -> {dest}')
print('Use train.json and dev.json ONLY for independent split evaluation.')
