# VeriGraph AI v0.6.1a — Chinese NER Annotation Review

## Tujuan
Memindahkan review kandidat NER Mandarin dari workbook ke **tab 8** Streamlit tanpa menjadikan rekomendasi AI sebagai label gold secara otomatis. Perubahan ini bersifat **additive**: tab KG/ML dan alur v0.6.0 tetap ada.

## Input dan sumber
- `data/pilots/cfever_zh_train_50.jsonl` — 50 klaim dari **CFEVER train**, seed 42; hanya teks klaim dan provenance.
- `data/annotations/cfever_zh_review_candidates_v061a.json` — snapshot **122 kandidat**: 114 pada `Draft_Gabungan_50` + 8 pada `Kandidat_Tambahan_Review`.
- Workbook asli review AI tahap 2 tersedia di `05_DOCUMENTATION` master sebagai dokumentasi provenance. AI memberikan 96 rekomendasi terima, 4 revisi, 12 tolak, dan 10 adjudikasi; **semuanya belum menjadi keputusan manusia**.
- Untuk tiga kandidat (`main:0014`, `main:0042`, `main:0070`), offset kolom workbook mengacu pada span **revisi AI** alih-alih span kandidat asal. Snapshot menyimpan `excel_start/excel_end`, offset kandidat asal yang dikoreksi melalui kemunculan teks, dan `ai_offsets` terpisah agar dapat diaudit. Keputusan tetap PENDING.

## Penggunaan
Di direktori `01_ACTIVE_PROJECT/verigraph_ai_starter`:
```cmd
conda activate kg_e
python -m pytest -q
python scripts\check_ner_annotation_v061a.py
python -m streamlit run app.py
```
Masuk ke tab **8. Chinese NER Review (v0.6.1a)**. Pilih klaim, tinjau AI suggestion, tentukan `ACCEPT`/`REVISE`/`REJECT`/`OUT_OF_SCOPE`/`NEEDS_CONTEXT`, dan **klik Simpan**. Untuk revisi, masukkan offset `start` (inklusif) dan `end` (eksklusif) terhadap teks Mandarin asli; aplikasi memberi pratinjau potongan teks. Untuk mention yang tidak diusulkan AI, gunakan *Tambah entitas yang terlewat*.

Setelah semua kandidat dalam sebuah klaim diputuskan, periksa teks aslinya sekali lagi lalu pilih `CONFIRMED_COMPLETE` jika ada entitas inti, atau `CONFIRMED_NO_CORE` jika tidak. Mengubah keputusan kandidat akan mereset status kelengkapan klaim untuk mencegah konfirmasi kedaluwarsa.

## Penyimpanan dan keamanan
- Keputusan manusia + riwayat setiap revisi: `outputs/ner_review_v061a/decisions.json`.
- Penulisan keputusan dilakukan secara **atomic** (file temporary pada folder yang sama + rename); pergantian kandidat atau review yang belum disimpan tidak akan disimpan otomatis.
- **Jangan menghapus file `decisions.json`.** Backup melalui tombol download di tab 8 dan/atau salin ke Drive.
- Data keputusan memiliki fingerprint snapshot kandidat sehingga keputusan dari versi anotasi berbeda tidak digabung otomatis.
- Pengguna yang memperbarui master lama harus menjalankan `PASANG_KE_MASTER_LOKAL.py`; installer membackup `app.py` dan file kode yang ditimpa serta tidak mengganti keputusan yang sudah ada.

## Kriteria siap diekspor
Dataset *reviewed* empat tipe (`person`, `organization`, `location`, `event`) hanya dapat diekspor jika:
1. Seluruh kandidat (termasuk tambahan manual) diputuskan eksplisit, tanpa PENDING/NEEDS_CONTEXT.
2. Seluruh **50 klaim** dikonfirmasi lengkap atau dikonfirmasi tanpa entitas inti.
3. Offset dan surface akhir valid terhadap teks klaim.
4. Tidak ada **overlap** antarmember anotasi inti dalam klaim yang sama (konvensi pilot *flat NER*).

Output diberi status `reviewed_for_evaluation_pending_independent_audit`, **bukan gold final**. Belum boleh mengklaim precision/recall/F1 akhir, melakukan Wikidata QID assignment otomatis, atau menggunakan label kebenaran klaim sebagai bagian dari NER/KG.

## Protokol NER pilot
- Kategori inti: PERSON, ORGANIZATION, LOCATION, EVENT (disimpan sebagai `person`, `organization`, `location`, `event`).
- OTHER:* dan AMBIGUOUS:* disimpan sebagai bahan audit tetapi **bukan empat tipe inti**.
- Prioritas 14 kasus ambigu/revisi AI (10 needs adjudication + 4 revise recommended).
- Kebijakan batas karakter, demonym, nested locations, gelar, nama karya, dan benda langit perlu dikonsistenkan serta disetujui anotator.
- Evaluation yang valid setelah audit hanya untuk pilot train 50 klaim. Generalisasi perlu sample terpisah dari dev, dengan kebocoran model-selection dihindari.
