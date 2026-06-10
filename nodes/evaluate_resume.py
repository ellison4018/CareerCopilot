import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import LLM_BASE_URL, LLM_API_KEY
from state import ResumeState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluate_resume.txt"
_llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model="qwen3.7-plus",
    temperature=0.3,
)


def evaluate_resume(state: ResumeState) -> dict:
    """LLM 节点：对 profile 按维度评分并给出改进建议。"""
    print("[3/5] 评估简历质量（LLM 调用中）...")
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.format(profile=json.dumps(state["profile"], ensure_ascii=False, indent=2))

    response = _llm.invoke([HumanMessage(content=prompt)])
    evaluation = _parse_json(response.content)

    return {"evaluation": evaluation}


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
