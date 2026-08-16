import json
import os
from pathlib import Path
import urllib.error
import urllib.request


DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ENV_FILE = Path(__file__).resolve().parent / ".env"


class AIInterpreterError(Exception):
    """智能解读调用失败时抛出的业务异常。"""


def _build_endpoint(base_url: str) -> str:
    """把 OpenAI 兼容 base_url 转换为 chat/completions endpoint。"""
    if base_url.replace("/", "").replace(":", "") == DEFAULT_BASE_URL.replace("/", "").replace(":", ""):
        base_url = DEFAULT_BASE_URL
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def _get_config_value(name: str, default: str | None = None) -> str | None:
    """读取配置，优先当前进程环境变量，再读取项目根目录 .env。"""
    value = os.getenv(name)
    if value:
        return value
    if not ENV_FILE.exists():
        return default
    try:
        for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, item_value = line.split("=", 1)
            if key.strip() == name:
                return item_value.strip().strip('"').strip("'") or default
    except OSError:
        pass
    return default


def generate_stock_interpretation(context: dict, model: str = DEFAULT_MODEL) -> str:
    """调用 DashScope OpenAI 兼容接口生成股票研究解读。"""
    api_key = _get_config_value("DASHSCOPE_API_KEY")
    if not api_key:
        raise AIInterpreterError("未检测到 DASHSCOPE_API_KEY，请先在项目 .env 文件中配置 API Key。")

    base_url = _get_config_value("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    endpoint = _build_endpoint(base_url)
    payload = {
        "model": _get_config_value("DASHSCOPE_MODEL", model),
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个A股短线研究助手，只基于用户提供的数据做客观解释。"
                    "不要给出买入、卖出、加仓、清仓等交易指令；不要使用必涨、必跌等确定性措辞。"
                    "输出中文，结构清晰，控制在400字以内。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请根据以下结构化数据生成一段智能解读，包含："
                    "1）预测结论摘要；2）主要依据；3）异常波动对可靠性的影响；"
                    "4）支撑压力观察；5）风险提示。\n\n"
                    f"{json.dumps(context, ensure_ascii=False, default=str)}"
                ),
            },
        ],
        "temperature": 0.3,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise AIInterpreterError(f"智能解读接口返回错误：HTTP {exc.code} {detail}") from exc
    except Exception as exc:
        raise AIInterpreterError(f"智能解读接口调用失败：{exc}") from exc

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise AIInterpreterError("智能解读接口返回格式异常。") from exc
