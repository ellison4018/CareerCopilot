from typing import TypedDict


class ResumeState(TypedDict):
    # 输入
    raw_input: str          # PDF 提取文本 or 粘贴文本
    input_type: str         # "pdf" | "text"

    # 简历分析层
    profile: dict           # 结构化技能/经历/教育
    evaluation: dict        # 各维度评分 + 改进建议
    clarify_questions: list # clarify 节点生成的问题
    clarify_answers: str    # 用户回答

    # 职位匹配层
    constraints: dict       # parse_feedback 提取的结构化约束（revise 循环时生效）
    matched_jobs: list      # 从职位库检索出的候选岗位
    ranked_jobs: list       # 排序后 + 匹配理由

    # 输出与反馈
    final_report: str       # Markdown 格式报告
    user_feedback: str      # 用户修改意见
    iteration: int          # 循环次数，防无限循环