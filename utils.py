import pandas as pd


def safe_float(value, default=None):
    """安全转换浮点数，转换失败时返回默认值。"""
    try:
        result = pd.to_numeric(value, errors="coerce")
        if pd.isna(result):
            return default
        return float(result)
    except Exception:
        return default


def format_number(value, digits: int = 2, suffix: str = "") -> str:
    """格式化数字，空值时显示暂无数据。"""
    number = safe_float(value)
    if number is None:
        return "暂无数据"
    return f"{number:.{digits}f}{suffix}"


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """从候选字段中找到 DataFrame 中存在的第一个字段。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """按映射重命名存在的字段，忽略不存在的字段。"""
    rename_map = {src: dst for src, dst in mapping.items() if src in df.columns}
    return df.rename(columns=rename_map).copy()


def friendly_error(prefix: str, exc: Exception) -> str:
    """把异常转换为适合页面展示的简短错误信息。"""
    return f"{prefix}：{exc}"
