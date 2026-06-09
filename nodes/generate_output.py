from state import ResumeState


def generate_output(state: ResumeState) -> dict:
    """模板节点：将 evaluation + ranked_jobs 渲染为 Markdown 报告。"""
    profile     = state.get("profile", {})
    evaluation  = state.get("evaluation", {})
    ranked_jobs = state.get("ranked_jobs", [])

    report = _render(profile, evaluation, ranked_jobs)
    return {"final_report": report}


def _render(profile: dict, evaluation: dict, ranked_jobs: list) -> str:
    lines = []

    # ── 标题 ────────────────────────────────────────────────────────────────
    name = profile.get("name") or "候选人"
    lines.append(f"# CareerCopilot 职业报告 — {name}\n")

    # ── 个人概览 ─────────────────────────────────────────────────────────────
    lines.append("## 个人概览\n")
    fields = [
        ("目标岗位", profile.get("target_role")),
        ("工作年限", profile.get("years_of_experience")),
        ("所在城市", profile.get("location")),
    ]
    for label, val in fields:
        if val is not None:
            lines.append(f"- **{label}**：{val}")
    skills = profile.get("skills", [])
    if skills:
        lines.append(f"- **核心技能**：{', '.join(skills)}")
    lines.append("")

    # ── 简历评估 ─────────────────────────────────────────────────────────────
    lines.append("## 简历评估\n")
    scores = evaluation.get("scores", {})
    overall = evaluation.get("overall_score")
    if scores:
        lines.append("| 维度 | 得分 |")
        lines.append("|------|------|")
        dim_labels = {
            "skill_relevance":    "技能相关性",
            "experience_quality": "经历质量",
            "expression":         "表达规范性",
        }
        for key, label in dim_labels.items():
            if key in scores:
                lines.append(f"| {label} | {scores[key]} / 10 |")
        if overall is not None:
            lines.append(f"| **综合得分** | **{overall} / 10** |")
        lines.append("")

    strengths = evaluation.get("strengths", [])
    if strengths:
        lines.append("**优势**\n")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    improvements = evaluation.get("improvements", [])
    if improvements:
        lines.append("**改进建议**\n")
        for item in improvements:
            dim = item.get("dimension", "")
            issue = item.get("issue", "")
            suggestion = item.get("suggestion", "")
            lines.append(f"- **{dim}**：{issue} → {suggestion}")
        lines.append("")

    summary = evaluation.get("summary")
    if summary:
        lines.append(f"> {summary}\n")

    # ── 推荐职位 ─────────────────────────────────────────────────────────────
    lines.append("## 推荐职位\n")
    if not ranked_jobs:
        lines.append("_暂无匹配职位。_\n")
    else:
        for i, job in enumerate(ranked_jobs, start=1):
            title     = job.get("job_title", job.get("title", "未知职位"))
            score     = job.get("match_score", "-")
            reasons   = job.get("match_reasons", [])
            gaps      = job.get("gaps", [])
            advice    = job.get("advice", "")

            lines.append(f"### {i}. {title}　（匹配度 {score}/10）\n")

            if reasons:
                lines.append("**匹配理由**\n")
                for r in reasons:
                    lines.append(f"- {r}")
                lines.append("")

            if gaps:
                lines.append("**待补强**\n")
                for g in gaps:
                    lines.append(f"- {g}")
                lines.append("")

            if advice:
                lines.append(f"**求职建议**：{advice}\n")

    # ── 页脚 ─────────────────────────────────────────────────────────────────
    lines.append("---")
    lines.append("_由 CareerCopilot 生成_")

    return "\n".join(lines)
