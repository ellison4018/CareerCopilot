from langgraph.graph import StateGraph, END

from state import ResumeState
from nodes.parse_input import parse_input
from nodes.extract_profile import extract_profile
from nodes.evaluate_resume import evaluate_resume
from nodes.clarify import clarify
from nodes.match_jobs import match_jobs
from nodes.rank_and_explain import rank_and_explain
from nodes.generate_output import generate_output
from nodes.revise import revise
from nodes.parse_feedback import parse_feedback


def should_clarify(state: ResumeState) -> str:
    """条件路由：profile 关键字段缺失则补充，否则直接匹配职位。"""
    profile = state.get("profile", {})
    missing = not profile.get("target_role") or profile.get("years_of_experience") is None
    return "clarify" if missing else "match_jobs"


def after_clarify(state: ResumeState) -> str:
    """clarify 后：有用户回答则重新提取 profile，否则直接匹配职位。"""
    return "extract_profile" if state.get("clarify_answers") else "match_jobs"


def human_feedback_router(state: ResumeState) -> str:
    """条件路由：首次始终征求反馈；后续仅在有实质反馈且未超迭代上限时循环。"""
    iteration = state.get("iteration", 0)
    feedback  = state.get("user_feedback", "").strip()

    # 第一次（iteration=0）：还没有征求过反馈，必须进入 revise
    if iteration == 0:
        return "revise"

    # 后续：用户给了实质反馈且未超上限 → 继续循环
    if feedback and iteration < 3:
        return "revise"

    # 用户跳过 或 达到上限 → 结束
    return END


def build_graph(checkpointer=None) -> StateGraph:
    builder = StateGraph(ResumeState)

    # 注册节点
    builder.add_node("parse_input", parse_input)
    builder.add_node("extract_profile", extract_profile)
    builder.add_node("evaluate_resume", evaluate_resume)
    builder.add_node("clarify", clarify)
    builder.add_node("match_jobs", match_jobs)
    builder.add_node("rank_and_explain", rank_and_explain)
    builder.add_node("generate_output", generate_output)
    builder.add_node("revise", revise)
    builder.add_node("parse_feedback", parse_feedback)

    # 入口
    builder.set_entry_point("parse_input")

    # 固定边
    builder.add_edge("parse_input", "extract_profile")
    builder.add_edge("extract_profile", "evaluate_resume")
    builder.add_conditional_edges(
        "clarify", after_clarify,
        {"extract_profile": "extract_profile", "match_jobs": "match_jobs"},
    )
    builder.add_edge("match_jobs", "rank_and_explain")
    builder.add_edge("rank_and_explain", "generate_output")
    builder.add_edge("revise", "parse_feedback")          # 反馈 → 提炼约束
    builder.add_edge("parse_feedback", "match_jobs")      # 约束 → 重新检索

    # 条件边
    builder.add_conditional_edges(
        "evaluate_resume",
        should_clarify,
        {"clarify": "clarify", "match_jobs": "match_jobs"},
    )
    builder.add_conditional_edges(
        "generate_output",
        human_feedback_router,
        {"revise": "revise", END: END},
    )

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
