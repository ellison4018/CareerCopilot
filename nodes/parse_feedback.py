import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import LLM_BASE_URL, LLM_API_KEY
from state import ResumeState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "parse_feedback.txt"
_llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model="qwen3.7-plus",
    temperature=0,
)


def parse_feedback(state: ResumeState) -> dict:
    """LLM 节点：将用户自然语言反馈提取为结构化检索约束。"""
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.format(feedback=state.get("user_feedback", ""))

    response = _llm.invoke([HumanMessage(content=prompt)])
    constraints = _parse_json(response.content)

    return {"constraints": constraints}


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}
