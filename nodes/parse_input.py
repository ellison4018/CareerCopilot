import re
import pymupdf

from state import ResumeState


def parse_input(state: ResumeState) -> dict:
    """纯工具节点：PDF 提取文本 or 清洗粘贴文本，统一输出 raw_input。"""
    input_type = state["input_type"]

    if input_type == "pdf":
        raw = _extract_pdf(state["raw_input"])
    else:
        raw = _clean_text(state["raw_input"])

    return {"raw_input": raw}


def _extract_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def _clean_text(text: str) -> str:
    text = text.strip()
    # 合并连续空行为单个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
