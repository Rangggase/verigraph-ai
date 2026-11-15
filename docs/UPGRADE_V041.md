# VeriGraph AI v0.4.1 — KG Quality & Enrichment

## Apa yang berbeda dari v0.4?

v0.4 adalah batch NER + graf `Claim -> MENTIONS -> Mention`. Hasil pengguna (5 klaim AVeriTeC) memiliki 23 node, 18 `MENTIONS`, 0 relasi semantik. v0.4.1 **mengolah berkas v0.4 yang sudah ada**, sehingga GLiNER *tidak perlu dijalankan ulang*.

v0.4.1 menambahkan:

1. **Quality flags** untuk mention berconfidence rendah dan kemungkinan jenis NER yang salah. Ini hanya peringatan heuristik, bukan gold annotations.
2. **Human review**: `pending`, `accepted_mention`, atau `rejected_mention` per mention. Seluruh keluaran GLiNER awal tetap tersimpan untuk audit.
3. **Entity identity resolution**: Wikidata ID (QID) hanya boleh dimasukkan setelah pengguna secara manual memeriksa kandidat. `RESOLVES_TO` membuat entitas sama bisa disatukan lintas klaim tanpa membuang mention aslinya.
4. **Candidate relations** dari aturan offline yang dapat diaudit atau *opsional* LLM Ollama lokal menggunakan JSON schema. Predikat, indeks entitas dan evidence quote diperiksa. Kandidat relasi **tidak pernah** berubah menjadi fakta otomatis.
5. **Graph export + quality report** dalam `outputs/batch_kg_v041.json`.

**Peringatan utama:** seluruh edge `claim_assertion` tetap `truth_status=unverified`; label, justifikasi, dan gold evidence dataset tidak diimpor ke KG. Mencocokkan QID tidak membuktikan kebenaran klaim. Graf ini belum merupakan verified factual KG / fact-checker atau model GNN.

## Cara pakai cepat — Windows, Conda

1. Ekstrak ZIP v0.4.1 ke direktori **baru**, buka folder `verigraph_ai_starter` di VS Code.
2. `conda activate kg_e`
3. `python -m pytest -q`
4. `python -m streamlit run app.py`
5. Buka tab **5. KG Quality & Enrichment (v0.4.1)**.
6. Pilih **Included 5-claim example**, atau copy hasil v0.4 milikmu ke `outputs/batch_kg_v04.json` lalu pilih **Saved v0.4 output**, atau pilih **Upload v0.4 JSON**.
7. Lihat quality flags, lalu tinjau kolom `decision`, `qid`, `canonical_label` di tabel review. Untuk mencari QID, pilih mention dan klik **Search Wikidata candidates**. Tidak ada pengaitan otomatis. Kamu boleh membiarkan QID kosong.
8. Pilih **Auditable rules (offline)** dan klik **Build reviewed candidate KG (v0.4.1)**.
9. Periksa quality report dan graf, lalu **Download v0.4.1 reviewed KG JSON**.

Pada lima klaim contoh, keluaran default offline sebelum review manual menghasilkan **18 mentions + 2 candidate relations** (`BOARD_MEMBER_OF` dan `PLEDGED_TO`) dan menandai beberapa mention berisiko. Angka ini adalah ukuran keberhasilan pipeline contoh, **bukan** akurasi atau precision model.

### CLI tanpa Streamlit (untuk debugging dan reproducibility)

Jalankan dari dalam root proyek (berisi `app.py`):

```cmd
python -m scripts.enrich_existing_batch --input data/examples/averitec_5_v04.json
```

Output:
- `outputs/batch_kg_v041.json`
- `outputs/mention_review_template.csv`

Kamu bisa mengedit CSV tersebut pada kolom `decision`, `qid`, dan `canonical_label`, kemudian mengulang proses:

```cmd
python -m scripts.enrich_existing_batch --input data/examples/averitec_5_v04.json --review-csv outputs/mention_review_template.csv
```

Tidak perlu mengubah `mention_key`, `claim_id`, `surface`, atau metadata hasil GLiNER. QID hanya diperbolehkan untuk `accepted_mention`, dan harus dicek secara manual pada Wikidata.

## Opsional: ekstraksi relasi dengan LLM lokal

Bila kamu sudah menginstal [Ollama](https://ollama.com/) dan model yang cukup ringan untuk perangkatmu:

```cmd
ollama pull qwen2.5:3b
ollama serve
```

Di terminal berbeda, jalankan Streamlit dan pilih **Local Ollama LLM (optional)**, atau CLI:

```cmd
python -m scripts.enrich_existing_batch --input data/examples/averitec_5_v04.json --extractor ollama --ollama-model qwen2.5:3b
```

Ollama berbicara dengan API lokal `http://localhost:11434/api/chat`. Default v0.4.1 tidak memakai layanan LLM eksternal dan tidak memerlukan API key. *Format JSON schema* membatasi bentuk respons, tetapi **tidak menjamin kebenaran semantik** hasil ekstraksi; review manual tetap dibutuhkan. Tidak ada model Ollama yang disertakan di ZIP. Pada CPU, inference LLM bisa lambat dan memakai banyak RAM.

## Kualitas NER pada contoh

- `20%` diprediksi `event` -> **numeric_as_event** (perlu koreksi).
- `American history` diprediksi `location` -> **abstract_as_location**.
- `In Nigeria` -> **leading_preposition** (sebaiknya normalisasi span setelah manual review).
- `illegal immigrants` diprediksi `organization` -> **generic_group_or_ambiguous_org**.

Quality flags hanya menandai; tidak menyunting otomatis output GLiNER. Mention berflag tidak digunakan untuk membentuk relasi semantik **sebelum** diterima secara manual. Aturan tersebut tidak menghalangi penyimpanan raw `MENTIONS` sebagai audit.

## Batasan yang sengaja dipertahankan

- Belum ada pengukuran NER F1 atau relation extraction precision pada data gold.
- Belum ada verified factual KG atau perbandingan dengan bukti independen.
- Evaluasi tiga bahasa belum lengkap; aturan offline lebih sempit dari kemampuan LLM.
- QID search adalah *candidate search*, bukan auto linker yang tepercaya.
- Pemrosesan massal dibatasi pada ukuran batch v0.4; kerjakan sampel dulu.
- Penyimpanan review lokal melalui JSON; belum ada database produksi.

## Langkah berikutnya (v0.5)

Setelah quality review pada sampel 30–50 klaim dan anotasi manual beberapa relasi, tambahkan dataset evidence terpisah dan baseline klasifikasi ML/DL dengan train/validation/test terkontrol. Bandingkan text-only dengan fitur graf; jangan mengklaim KG meningkatkan deteksi sebelum ada evaluasi.
