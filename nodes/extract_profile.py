import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import LLM_BASE_URL, LLM_API_KEY
from state import ResumeState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extract_profile.txt"
_llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model="qwen3.7-plus",
    temperature=0.1,
)


def extract_profile(state: ResumeState) -> dict:
    """LLM 节点：将原始简历文本结构化为 profile JSON。"""
    print("[2/5] 提取简历结构（LLM 调用中）...")
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    # 如果有 clarify 补充信息，注入到 prompt
    clarify_context = ""
    if state.get("clarify_answers"):
        clarify_context = f"Additional information provided by the user:\n{state['clarify_answers']}"

    prompt = prompt_template.format(
        resume_text=state["raw_input"],
        clarify_context=clarify_context,
    )

    response = _llm.invoke([HumanMessage(content=prompt)])
    profile = _parse_json(response.content)

    return {"profile": profile}


def _parse_json(text: str) -> dict:
    text = text.strip()
    # 兼容模型偶尔输出 ```json ... ``` 的情况
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
