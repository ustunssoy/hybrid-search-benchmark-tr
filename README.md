# Hibrit Arama Benchmark (BM25 + Dense)

**Python, FAISS, Sentence-Transformers, rank-bm25**

BM25, dense ve hibrit aramayı Türkçe MS MARCO üzerinde karşılaştırır. Skorları Reciprocal Rank Fusion (RRF) ile birleştirir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python benchmark.py
```

## Yöntemler

| Yöntem | Açıklama |
|--------|----------|
| BM25 | TF-IDF tabanlı leksikal arama (rank-bm25) |
| Dense | Transformer embedding + FAISS cosine similarity |
| Hibrit | RRF ile BM25 + Dense skor birleştirme |

## Sonuçlar

| Metrik | BM25 | Dense | Hibrit |
|--------|------|-------|--------|
| MRR@10 | 0.28 | 0.33 | **0.38** |
| nDCG@10 | 0.34 | 0.40 | **0.45** |
| Recall@10 | 0.38 | 0.44 | **0.52** |

Hibrit yöntem her metrikte en iyi sonucu verir.

## Yapı

```
├── benchmark.py       # Ana benchmark scripti
├── requirements.txt   # Bağımlılıklar
└── data/
    └── results.json   # Çıktı sonuçları
```

## Model

- **Dense:** `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr`
- **Veri:** `unicamp-dl/mmarco` (Turkish split)

## Lisans

MIT
