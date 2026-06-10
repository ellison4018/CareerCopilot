"""
CareerCopilot CLI 入口

用法:
  python main.py resume.pdf
  python main.py --text
"""
import sys
import argparse
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph import build_graph


def parse_args():
    parser = argparse.ArgumentParser(description="CareerCopilot — 简历评估与职位推荐")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pdf", nargs="?", help="PDF 简历路径")
    group.add_argument("--text", action="store_true", help="粘贴文本简历")
    return parser.parse_args()


def get_input(args) -> tuple[str, str]:
    """返回 (raw_input, input_type)"""
    if args.text:
        print("请粘贴简历文本，输入完成后按 Enter 再输入 END 结束：")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        return "\n".join(lines), "text"
    else:
        pdf_path = args.pdf
        if not Path(pdf_path).exists():
            print(f"文件不存在：{pdf_path}", file=sys.stderr)
            sys.exit(1)
        return pdf_path, "pdf"


def run():
    args = parse_args()
    raw_input, input_type = get_input(args)

    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "raw_input":  raw_input,
        "input_type": input_type,
        "profile":    {},
        "evaluation": {},
        "clarify_questions": [],
        "clarify_answers":   "",
        "constraints":   {},
        "matched_jobs":  [],
        "ranked_jobs":   [],
        "final_report":  "",
        "user_feedback": "",
        "iteration":     0,
    }

    print("\n── 开始分析简历 ──\n")

    # ── 主执行循环，处理所有 interrupt 事件 ──────────────────────────────────
    result = graph.invoke(initial_state, config=config)

    while True:
        # 检查是否有待处理的 interrupt
        state_snapshot = graph.get_state(config)
        pending = state_snapshot.tasks

        if not pending:
            # 没有 interrupt，graph 已运行到终点
            break

        # 取第一个 interrupt 的 payload
        interrupt_payload = None
        for task in pending:
            for interrupt in getattr(task, "interrupts", []):
                interrupt_payload = interrupt.value
                break
            if interrupt_payload:
                break

        if interrupt_payload is None:
            break

        # 根据 payload 内容判断是哪个 interrupt 节点
        message = interrupt_payload.get("message", "")
        print(f"\n{message}")

        if "questions" in interrupt_payload:
            # clarify 节点
            print(interrupt_payload["questions"])
            answer = input("\n您的回答：").strip()
        else:
            # revise 节点：先打印当前报告再询问
            report = interrupt_payload.get("current_report", "")
            if report:
                print("\n" + "═" * 60)
                print(report)
                print("═" * 60)
            answer = input("\n您的意见（直接回车跳过）：").strip()
            if not answer:
                # 用户跳过反馈，直接退出循环
                break

        # 用用户输入恢复 graph
        result = graph.invoke(Command(resume=answer), config=config)

    # ── 输出最终报告 ─────────────────────────────────────────────────────────
    final_state = graph.get_state(config).values
    report = final_state.get("final_report", "")

    if report:
        print("\n" + "═" * 60)
        print(report)
        print("═" * 60)
    else:
        print("\n未能生成报告，请检查输入。", file=sys.stderr)


if __name__ == "__main__":
    run()
