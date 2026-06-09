"""
构建检索索引：Chroma 向量库 + BM25 pickle
用法: python scripts/build_index.py

输出:
  index/chroma/        Chroma 持久化目录
  index/bm25.pkl       BM25 索引 + doc_ids 映射
"""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # 确保 config.py 可被 import

JOBS_PATH = ROOT / "jobs" / "jobs.json"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH  = INDEX_DIR / "bm25.pkl"

CHROMA_COLLECTION = "jobs"
EMBED_BATCH_SIZE = 10  # DashScope text-embedding-v4 单批最多 10 条


def build_doc_text(job: dict) -> str:
    """拼接用于 BM25 的检索文本。"""
    parts = [
        job.get("title", ""),
        job.get("position_name", ""),
        job.get("industry", ""),
        " ".join(job.get("skills", [])),
        job.get("description", ""),
    ]
    return " ".join(p for p in parts if p)


def build_embed_text(job: dict) -> str:
    """拼接用于向量嵌入的文本（description + skills 已足够语义丰富）。"""
    skills = " ".join(job.get("skills", []))
    desc   = job.get("description", "")
    title  = job.get("title", "")
    return f"{title}\n技能：{skills}\n{desc}"


def build_chroma(jobs: list[dict]) -> None:
    import chromadb
    from config import embedding_model

    print(f"[Chroma] 初始化持久化目录: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 重建时先删旧集合
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("[Chroma] 已删除旧集合，重新构建")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    ids       = [str(job["id"]) for job in jobs]
    texts     = [build_embed_text(job) for job in jobs]
    metadatas = [
        {
            "title":          job.get("title", ""),
            "location":       job.get("location", ""),
            "district":       job.get("district", ""),
            "years_required": job.get("years_required", 0),
            "salary_min":     job.get("salary_min", 0),
            "salary_max":     job.get("salary_max", 0),
            "degree_required": job.get("degree_required") or "",
        }
        for job in jobs
    ]

    # 分批 embed 并写入
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        end   = min(start + EMBED_BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_ids   = ids[start:end]
        batch_meta  = metadatas[start:end]

        embeddings = embedding_model.embed_documents(batch_texts)
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        print(f"[Chroma] 写入 {end}/{total}", end="\r")

    print(f"\n[Chroma] 完成，共 {collection.count()} 条向量")


def build_bm25(jobs: list[dict]) -> None:
    from rank_bm25 import BM25Okapi

    doc_ids = [str(job["id"]) for job in jobs]
    corpus  = [build_doc_text(job).split() for job in jobs]  # 简单空格分词

    bm25 = BM25Okapi(corpus)
    payload = {"bm25": bm25, "doc_ids": doc_ids}

    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[BM25]  完成，共 {len(doc_ids)} 条文档 → {BM25_PATH}")


def main():
    if not JOBS_PATH.exists():
        print(f"jobs.json 不存在: {JOBS_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(JOBS_PATH, encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"加载 {len(jobs)} 条职位数据")

    INDEX_DIR.mkdir(exist_ok=True)

    build_bm25(jobs)
    build_chroma(jobs)

    print("\n索引构建完成！")
    print(f"  Chroma: {CHROMA_DIR}")
    print(f"  BM25:   {BM25_PATH}")


if __name__ == "__main__":
    main()
