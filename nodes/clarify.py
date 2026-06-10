from langgraph.types import interrupt

from state import ResumeState


def clarify(state: ResumeState) -> dict:
    """Human-in-the-loop 节点：生成追问并暂停 graph 等待用户输入。"""
    profile = state.get("profile", {})

    missing = []
    if not profile.get("target_role"):
        missing.append("您的目标岗位是什么？（例如：后端工程师、数据分析师）")
    if profile.get("years_of_experience") is None:
        missing.append("您的工作年限大约是多少年？")

    if not missing:
        return {"clarify_questions": [], "clarify_answers": ""}

    questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(missing))

    answer = interrupt({
        "questions": questions,
        "message": "请回答以下问题以便更准确地为您推荐职位：",
    })

    return {
        "clarify_questions": missing,
        "clarify_answers": answer,
    }
