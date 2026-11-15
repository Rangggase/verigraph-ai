# Arsitektur VeriGraph AI

## Alur sistem

```mermaid
flowchart TD
  U[Pengguna] --> APP[app.py Streamlit]
  APP -->|mode mudah| EASY[easy_home_ui + ner_annotation_ui]
  APP -->|mode peneliti| TABS[8 tab]
  TABS --> BP[batch_pipeline]
  BP --> NER[ner_pipeline GLiNER]
  BP --> REL[relation_extraction]
  TABS --> QE[kg_enrichment]
  QE --> RG[reviewed_graph]
  RG --> OUT[(outputs/*.json)]
  EASY --> ST[(decisions.json atomik)]
  TABS --> CLS[classification_ui]
  CLS --> MI[model_integrity SHA-256]
  MI --> MODEL[(model.joblib)]
  TABS -. kandidat saja .-> WD[Wikidata API]
```

## Alur proses review entitas (bisnis)

```mermaid
sequenceDiagram
  participant R as Reviewer
  participant UI as Review NER
  participant S as State JSON
  UI->>R: Tampilkan kalimat + nama yang disorot
  R->>UI: Terima / Revisi / Tolak / Perlu konteks
  UI->>UI: validate_span (offset, tipe)
  UI->>S: Simpan atomik + event audit + revision++
  R->>UI: Tandai kalimat selesai
  UI->>S: claim_reviews diperbarui
  UI-->>R: Ringkasan progres
```

## Model data (ERD logis, JSON)

Tidak ada database relasional; entitas di bawah adalah struktur dokumen JSON yang divalidasi di kode.

```mermaid
erDiagram
  CLAIM ||--o{ MENTION : memuat
  MENTION ||--o| QID_LINK : "disetujui manusia"
  MENTION ||--o{ RELATION_CANDIDATE : "subjek/objek"
  CLAIM ||--o{ RELATION_CANDIDATE : evidence_text
  CLAIM { string claim_id PK
          string text
          string language
          string truth_status "selalu unverified" }
  MENTION { string mention_key PK
            string surface
            string ner_label
            int start
            int end
            string decision "pending|accepted|rejected" }
  QID_LINK { string qid "Q[1-9][0-9]*"
             string canonical_label }
  RELATION_CANDIDATE { string predicate
                       string decision "pending|accepted_extraction|rejected_extraction" }
```

## Invarian integritas

| Invarian | Penegak |
|---|---|
| Node klaim dan semua edge `unverified` | `reviewed_graph.load_v041` |
| Tidak ada `gold_label`, `evidence_json`, `justification` di graf | `PROHIBITED_FIELDS` |
| Edge tidak boleh menunjuk node yang tidak ada | validasi impor sebelum NetworkX |
| Candidate ID dan span unik, surface cocok dengan offset | `ner_annotation.load_candidates` |
| File keputusan hanya cocok dengan snapshot sumbernya | `source_digest` |
| Train dan dev tidak beririsan | `classification_data.verify_disjoint` |
| Model hanya dimuat bila SHA-256 cocok | `model_integrity.verify_model` |

## Sistem desain

Token warna/tipografi/spasi di `src/theme.py`; semua pasangan teks-latar diuji kontrasnya (WCAG AA). Penanda "belum terverifikasi" memakai garis putus-putus amber, sejajar dengan edge putus-putus untuk asersi di tampilan graf.
