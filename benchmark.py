import json
import numpy as np
import faiss
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── Veri Yükleme ──

def load_ms_marco_tr(split="train", max_queries=500, max_corpus=50000):
    ds = load_dataset("unicamp-dl/mmarco", "turkish", split=split, streaming=True, trust_remote_code=True)

    corpus = {}
    queries = {}
    qrels = {}
    seen_qids = set()

    for row in tqdm(ds, desc="Veri yükleniyor"):
        qid = str(row["query_id"])
        pid_pos = str(row["positive_id"])
        pid_neg = str(row["negative_id"])

        if qid not in seen_qids and len(queries) < max_queries:
            queries[qid] = row["query"]
            seen_qids.add(qid)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid_pos] = 1

        if pid_pos not in corpus and len(corpus) < max_corpus:
            corpus[pid_pos] = row["positive"]
        if pid_neg not in corpus and len(corpus) < max_corpus:
            corpus[pid_neg] = row["negative"]

        if len(queries) >= max_queries and len(corpus) >= max_corpus:
            break

    return corpus, queries, qrels


# ── BM25 ──

def build_bm25_index(corpus_texts):
    tokenized = [doc.lower().split() for doc in corpus_texts]
    return BM25Okapi(tokenized)


def bm25_search(bm25, query, corpus_ids, top_k=100):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(corpus_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


# ── Dense ──

def build_dense_index(corpus_texts, model, batch_size=256):
    embeddings = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def dense_search(model, index, query, corpus_ids, top_k=100):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, top_k)
    return [(corpus_ids[idx], float(scores[0][i])) for i, idx in enumerate(indices[0]) if idx != -1]


# ── RRF ──

def reciprocal_rank_fusion(result_lists, k=60):
    fused = {}
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ── Metrikler ──

def mrr_at_k(results, qrels, k=10):
    for rank, (doc_id, _) in enumerate(results[:k]):
        if doc_id in qrels and qrels[doc_id] > 0:
            return 1.0 / (rank + 1)
    return 0.0


def ndcg_at_k(results, qrels, k=10):
    dcg = 0.0
    for rank, (doc_id, _) in enumerate(results[:k]):
        rel = qrels.get(doc_id, 0)
        dcg += rel / np.log2(rank + 2)

    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(results, qrels, k=10):
    retrieved = {doc_id for doc_id, _ in results[:k]}
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


# ── Ana Akış ──

def main():
    print("=" * 60)
    print("Hibrit Arama Benchmark (BM25 + Dense)")
    print("=" * 60)

    print("\n[1/5] MS MARCO-TR yükleniyor...")
    corpus, queries, qrels = load_ms_marco_tr(max_queries=500, max_corpus=50000)

    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())

    print(f"  Sorgu: {len(queries)}, Korpus: {len(corpus)}")

    print("\n[2/5] BM25 indeksi oluşturuluyor...")
    bm25 = build_bm25_index(corpus_texts)

    print("\n[3/5] Dense model yükleniyor...")
    model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

    print("\n[4/5] FAISS indeksi oluşturuluyor...")
    dense_index = build_dense_index(corpus_texts, model)

    print("\n[5/5] Değerlendirme...")
    metrics = {"bm25": {"mrr@10": [], "ndcg@10": [], "recall@10": []},
               "dense": {"mrr@10": [], "ndcg@10": [], "recall@10": []},
               "hybrid": {"mrr@10": [], "ndcg@10": [], "recall@10": []}}

    for qid, query in tqdm(queries.items(), desc="Sorgular değerlendiriliyor"):
        if qid not in qrels:
            continue
        qrel = qrels[qid]

        bm25_results = bm25_search(bm25, query, corpus_ids)
        dense_results = dense_search(model, dense_index, query, corpus_ids)
        hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results])

        for method, results in [("bm25", bm25_results), ("dense", dense_results), ("hybrid", hybrid_results)]:
            metrics[method]["mrr@10"].append(mrr_at_k(results, qrel))
            metrics[method]["ndcg@10"].append(ndcg_at_k(results, qrel))
            metrics[method]["recall@10"].append(recall_at_k(results, qrel))

    # ── Sonuçlar ──
    print("\n" + "=" * 60)
    print(f"{'Metrik':<12} {'BM25':>10} {'Dense':>10} {'Hibrit':>10}")
    print("-" * 42)

    results_dict = {}
    for metric_name in ["mrr@10", "ndcg@10", "recall@10"]:
        bm25_val = np.mean(metrics["bm25"][metric_name])
        dense_val = np.mean(metrics["dense"][metric_name])
        hybrid_val = np.mean(metrics["hybrid"][metric_name])
        print(f"{metric_name:<12} {bm25_val:>10.4f} {dense_val:>10.4f} {hybrid_val:>10.4f}")
        results_dict[metric_name] = {"bm25": bm25_val, "dense": dense_val, "hybrid": hybrid_val}

    print("=" * 60)

    with open(DATA_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({k: {m: float(v) for m, v in vals.items()} for k, vals in results_dict.items()}, f, indent=2)

    print(f"\nSonuçlar {DATA_DIR / 'results.json'} dosyasına kaydedildi.")


if __name__ == "__main__":
    main()
