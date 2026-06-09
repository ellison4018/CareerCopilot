"""
将原始 CSV 转换为 jobs/jobs.json
用法: python scripts/csv_to_json.py
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "sw_job_202606091504.csv"
OUT_PATH = ROOT / "jobs" / "jobs.json"

# experience 字段编码 → 最低工作年限
EXP_MAP = {
    "0":    0,   # 不限
    "2003": 1,   # 1年以上
    "2004": 3,   # 3年以上
    "2005": 5,   # 5年以上
    "2006": 10,  # 10年以上
}

# degree 字段编码 → 学历要求
DEG_MAP = {
    "0":    None,
    "1001": "初中",
    "1002": "高中",
    "1003": "大专",
    "1004": "本科",
    "1005": "硕士",
    "1006": "博士",
}


def parse_salary(s: str) -> tuple[int, int]:
    """'8-12K' → (8000, 12000)，解析失败返回 (0, 0)"""
    s = s.strip().upper()
    multiplier = 10000 if "W" in s else 1000
    s = s.replace("K", "").replace("W", "")
    if "-" in s:
        parts = s.split("-")
        try:
            return int(parts[0]) * multiplier, int(parts[1]) * multiplier
        except ValueError:
            return 0, 0
    return 0, 0


def parse_json_field(raw: str) -> list:
    try:
        return json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return []


def convert(row: dict) -> dict:
    salary_min, salary_max = parse_salary(row["salary"])
    return {
        "id":               row["id"],
        "title":            row["name"],
        "position_name":    row["position_name"],
        "industry":         row["industry_name"],
        "location":         row["region_name"],
        "district":         row["area_district_name"],
        "salary":           row["salary"],
        "salary_min":       salary_min,
        "salary_max":       salary_max,
        "years_required":   EXP_MAP.get(row["experience"], 0),
        "degree_required":  DEG_MAP.get(row["degree"]),
        "skills":           parse_json_field(row["skills"]),
        "welfares":         parse_json_field(row["welfares"]),
        "description":      row["description"].strip(),
    }


def main():
    if not CSV_PATH.exists():
        print(f"CSV 文件不存在: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        jobs = [convert(row) for row in reader if row.get("status") == "1"]

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)

    print(f"转换完成：{len(jobs)} 条职位 → {OUT_PATH}")


if __name__ == "__main__":
    main()
