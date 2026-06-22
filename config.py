import json
from pathlib import Path


CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG = {
    "default_period": 250,
    "default_pool": "自选股",
    "show_macd": True,
    "show_rsi": True,
    "show_boll": True,
}


def load_config(path: Path = CONFIG_FILE) -> dict:
    """读取系统设置，文件不存在或损坏时返回默认配置。"""
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_CONFIG.copy()
    if not isinstance(data, dict):
        return DEFAULT_CONFIG.copy()
    result = DEFAULT_CONFIG.copy()
    result.update({key: data.get(key, value) for key, value in DEFAULT_CONFIG.items()})
    return result


def save_config(config: dict, path: Path = CONFIG_FILE) -> None:
    """保存系统设置到本地 JSON 文件。"""
    result = DEFAULT_CONFIG.copy()
    result.update({key: config.get(key, value) for key, value in DEFAULT_CONFIG.items()})
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
