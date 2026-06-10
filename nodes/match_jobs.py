"""
match_jobs — 三层检索节点（纯工具，无 LLM 生成）

流程：
  1. 规则预过滤（工作年限 / 城市）
  2. Chroma 向量检索 Top-20
  3. BM25 关键词检索 Top-20
  4. RRF 融合 → Top-20
  5. Rerank 精排 → Top-10
"""
import json
import pickle
import sys
from pathlib import Path

import chromadb
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    RERANK_BASE_URL,
    RERANK_API_KEY,
    RERANK_MODEL,
    embedding_model,
)
from state import ResumeState

_JOBS_PATH   = ROOT / "jobs" / "jobs.json"
_CHROMA_DIR  = ROOT / "index" / "chroma"
_BM25_PATH   = ROOT / "index" / "bm25.pkl"

_CHROMA_COLLECTION = "jobs"
_VECTOR_TOP_K  = 20
_BM25_TOP_K    = 20
_RRF_K         = 60   # RRF 常数，标准取 60
_RERANK_TOP_N  = 10


# ── 懒加载单例，首次调用节点时初始化 ────────────────────────────────────────

_jobs_by_id: dict | None = None
_chroma_col = None
_bm25_payload: dict | None = None


def _load_resources():
    global _jobs_by_id, _chroma_col, _bm25_payload

    if _jobs_by_id is None:
        with open(_JOBS_PATH, encoding="utf-8") as f:
            jobs = json.load(f)
        _jobs_by_id = {str(j["id"]): j for j in jobs}

    if _chroma_col is None:
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        _chroma_col = client.get_collection(_CHROMA_COLLECTION)

    if _bm25_payload is None:
        with open(_BM25_PATH, "rb") as f:
            _bm25_payload = pickle.load(f)


# ── 1. 规则预过滤 ────────────────────────────────────────────────────────────

def _rule_filter(jobs: dict, profile: dict, constraints: dict) -> set[str]:
    """返回通过规则过滤的 job_id 集合。constraints 来自 parse_feedback，首次为空。"""
    years = profile.get("years_of_experience") or 0

    # 城市：constraints 优先，其次 profile
    location = (constraints.get("location") or profile.get("location") or "").strip()

    salary_min       = constraints.get("salary_min")           # 最低薪资（元）
    exclude_company  = (constraints.get("exclude_company_type") or "").strip()

    print(f"  [规则过滤] 工作年限≤{years}, 城市={location or '不限'}, "
          f"最低薪资={salary_min or '不限'}, 排除行业={exclude_company or '无'}")

    passed = set()
    for jid, job in jobs.items():
        if job.get("years_required", 0) > years:
            continue
        if location and job.get("location") and location not in job["location"]:
            continue
        if salary_min and job.get("salary_max", 0) < salary_min:
            continue
        if exclude_company and exclude_company in (job.get("industry") or ""):
            continue
        passed.add(jid)

    print(f"  [规则过滤] 原始职位 {len(jobs)} → 过滤后 {len(passed)}")
    return passed


# ── 2. Chroma 向量检索 ───────────────────────────────────────────────────────

def _vector_search(query: str, allowed_ids: set[str]) -> list[str]:
    query_vec = embedding_model.embed_query(query)
    results = _chroma_col.query(
        query_embeddings=[query_vec],
        n_results=min(_VECTOR_TOP_K, len(allowed_ids) or 1),
        where={"$and": []} if not allowed_ids else None,
        include=["distances"],
    )
    ids = results["ids"][0]
    matched = [jid for jid in ids if jid in allowed_ids][:_VECTOR_TOP_K]
    print(f"  [向量检索] 命中 {len(matched)} 条")
    return matched


# ── 3. BM25 关键词检索 ───────────────────────────────────────────────────────

def _bm25_search(query: str, allowed_ids: set[str]) -> list[str]:
    bm25    = _bm25_payload["bm25"]
    doc_ids = _bm25_payload["doc_ids"]

    scores = bm25.get_scores(query.split())
    ranked = sorted(
        ((doc_ids[i], scores[i]) for i in range(len(doc_ids)) if doc_ids[i] in allowed_ids),
        key=lambda x: x[1],
        reverse=True,
    )
    matched = [jid for jid, _ in ranked[:_BM25_TOP_K]]
    print(f"  [BM25检索] 命中 {len(matched)} 条")
    return matched


# ── 4. RRF 融合 ──────────────────────────────────────────────────────────────

def _rrf_merge(list1: list[str], list2: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for rank, jid in enumerate(list1, start=1):
        scores[jid] = scores.get(jid, 0.0) + 1.0 / (_RRF_K + rank)
    for rank, jid in enumerate(list2, start=1):
        scores[jid] = scores.get(jid, 0.0) + 1.0 / (_RRF_K + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── 5. Rerank 精排 ───────────────────────────────────────────────────────────

def _rerank(query: str, candidates: list[str]) -> list[str]:
    docs = [_jobs_by_id[jid]["title"] + "\n" + _jobs_by_id[jid]["description"][:500]
            for jid in candidates]

    resp = requests.post(
        RERANK_BASE_URL.rstrip('/'),
        headers={
            "Authorization": f"Bearer {RERANK_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      RERANK_MODEL,
            "input": {
                "query":     query,
                "documents": docs,
            },
            "parameters": {
                "top_n":          _RERANK_TOP_N,
                "return_documents": False,
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    # DashScope rerank 返回 output.results[].index
    results = data.get("output", {}).get("results", [])
    ranked_ids = [candidates[r["index"]] for r in results[:_RERANK_TOP_N]]

    print(f"  [Rerank] 请求 {RERANK_BASE_URL.rstrip('/')}")
    print(f"  [Rerank] 候选 {len(candidates)} → 精排 {len(ranked_ids)}")
    for i, jid in enumerate(ranked_ids, 1):
        title = _jobs_by_id[jid].get("title", _jobs_by_id[jid].get("job_title", "未知"))
        score = next((r.get("relevance_score", "?") for r in results if r["index"] == candidates.index(jid)), "?")
        print(f"    {i}. {title}  (score={score})")

    return ranked_ids


# ── 主节点函数 ───────────────────────────────────────────────────────────────

def match_jobs(state: ResumeState) -> dict:
    iteration = state.get("iteration", 0)
    print(f"\n[4/5] 检索匹配职位（向量 + BM25 + Rerank）  [迭代 #{iteration}]")
    _load_resources()

    profile     = state.get("profile", {})
    evaluation  = state.get("evaluation", {})
    constraints = state.get("constraints") or {}

    # 构建检索 query：target_role + 核心技能
    target     = constraints.get("target_role") or profile.get("target_role", "")
    skills_str = " ".join(profile.get("skills", [])[:10])
    query      = f"{target} {skills_str}".strip()

    # evaluation summary 增强语义
    summary = evaluation.get("summary", "")
    if summary:
        query = f"{query} {summary}"

    # constraints 补充关键词（来自用户反馈）
    query_append = (constraints.get("query_append") or "").strip()
    if query_append:
        query = f"{query} {query_append}"

    print(f"  [检索Query] {query}")

    allowed_ids = _rule_filter(_jobs_by_id, profile, constraints)
    if not allowed_ids:
        allowed_ids = set(_jobs_by_id.keys())  # 规则过滤后为空则不过滤

    vec_ids  = _vector_search(query, allowed_ids)
    bm25_ids = _bm25_search(query, allowed_ids)
    merged   = _rrf_merge(vec_ids, bm25_ids)

    print(f"  [RRF融合] 向量{len(vec_ids)} + BM25{len(bm25_ids)} → 融合 {len(merged)} 条")

    top_candidates = merged[:_RERANK_TOP_N * 2]  # 给 rerank 留足候选
    ranked_ids     = _rerank(query, top_candidates)

    matched_jobs = [_jobs_by_id[jid] for jid in ranked_ids if jid in _jobs_by_id]
    return {"matched_jobs": matched_jobs}
