import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import LLM_BASE_URL, LLM_API_KEY
from state import ResumeState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "rank_and_explain.txt"
_llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model="qwen3.7-plus",  # 推理能力更强，适合综合判断
    temperature=0.3,
)

# 传给 LLM 的 job 字段，裁剪掉无关的数值字段
_JOB_FIELDS = ("id", "title", "position_name", "industry", "location",
               "salary", "years_required", "degree_required", "skills", "description")


def rank_and_explain(state: ResumeState) -> dict:
    """LLM 节点：对 Top-10 候选岗位重排序并生成匹配理由 + 差距提示。"""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    jobs_slim = [
        {k: job[k] for k in _JOB_FIELDS if k in job}
        for job in state.get("matched_jobs", [])
    ]

    feedback = state.get("user_feedback", "").strip()
    feedback_section = (
        f"\nUser Feedback (apply these constraints to your selection):\n{feedback}\n"
        if feedback else ""
    )

    prompt = prompt_template.format(
        profile=json.dumps(state.get("profile", {}), ensure_ascii=False, indent=2),
        evaluation=json.dumps(state.get("evaluation", {}), ensure_ascii=False, indent=2),
        jobs=json.dumps(jobs_slim, ensure_ascii=False, indent=2),
    ) + feedback_section

    response = _llm.invoke([HumanMessage(content=prompt)])
    ranked_jobs = _parse_json(response.content)

    return {"ranked_jobs": ranked_jobs}


def _parse_json(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []
