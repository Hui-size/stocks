import json
from pathlib import Path

from data_fetch import DataFetchError, normalize_stock_code


WATCHLIST_FILE = Path("watchlist.json")


def load_watchlist(path: Path = WATCHLIST_FILE) -> list[str]:
    """从本地 JSON 文件读取自选股列表，文件不存在时返回空列表。"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    codes = []
    for item in data:
        try:
            code = normalize_stock_code(str(item))
            if code not in codes:
                codes.append(code)
        except DataFetchError:
            continue
    return codes


def save_watchlist(codes: list[str], path: Path = WATCHLIST_FILE) -> None:
    """把自选股列表保存到本地 JSON 文件。"""
    clean_codes = []
    for item in codes:
        code = normalize_stock_code(item)
        if code not in clean_codes:
            clean_codes.append(code)
    path.write_text(json.dumps(clean_codes, ensure_ascii=False, indent=2), encoding="utf-8")


def add_to_watchlist(stock_code: str, path: Path = WATCHLIST_FILE) -> tuple[bool, str]:
    """添加股票到自选股列表，返回是否成功和提示文本。"""
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        return False, str(exc)

    codes = load_watchlist(path)
    if code in codes:
        return False, f"{code} 已经在自选股列表中。"
    codes.append(code)
    save_watchlist(codes, path)
    return True, f"已添加 {code} 到自选股。"


def remove_from_watchlist(stock_code: str, path: Path = WATCHLIST_FILE) -> tuple[bool, str]:
    """从自选股列表删除股票，返回是否成功和提示文本。"""
    try:
        code = normalize_stock_code(stock_code)
    except DataFetchError as exc:
        return False, str(exc)

    codes = load_watchlist(path)
    if code not in codes:
        return False, f"{code} 不在自选股列表中。"
    codes.remove(code)
    save_watchlist(codes, path)
    return True, f"已删除 {code}。"
