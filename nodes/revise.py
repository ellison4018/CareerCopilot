from langgraph.types import interrupt

from state import ResumeState


def revise(state: ResumeState) -> dict:
    """Human-in-the-loop 节点：收集用户反馈，注入 state，iteration +1。"""
    result = interrupt({
        "message": "对以上推荐结果是否有调整意见？（例如：不要外企 / 只看深圳 / 薪资要求 20K 以上）",
        "current_report": state.get("final_report", ""),
    })

    return {
        "user_feedback": result,
        "iteration": state.get("iteration", 0) + 1,
    }
