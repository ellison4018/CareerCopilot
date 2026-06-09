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
    missing = not profile.get("target_role") or not profile.get("years_of_experience")
    return "clarify" if missing else "match_jobs"


def human_feedback_router(state: ResumeState) -> str:
    """条件路由：有用户反馈且未超过迭代上限则重新排序，否则结束。"""
    feedback = state.get("user_feedback", "").strip()
    iteration = state.get("iteration", 0)
    if feedback and iteration < 3:
        return "revise"
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
    builder.add_edge("clarify", "extract_profile")        # 补充信息后重新提取
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
