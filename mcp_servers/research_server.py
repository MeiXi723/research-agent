"""参考 MCP server：纯函数零网络，用作 MCP 层集成测试夹具与第三方 server 接入演示。

mcp 2.3.0 起官方 SDK 将 FastMCP 改名为 MCPServer（mcp.server.mcpserver），
本模块按 2.x API 编写；`run()` 默认 transport='stdio'，与 stdio_client 子进程拉起方式配套。

依赖说明：Windows 无系统时区数据库，zoneinfo 需要 tzdata 包（pip install tzdata，
零依赖纯数据包），否则任何 ZoneInfo 键都会 ZoneInfoNotFoundError。
"""

import json
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="research", instructions="研究助手辅助工具集：文本统计、单位换算、时区时间。")

# 换算因子与偏移写死，避免精度谜团；换算表只允许表内键对，表外一律报错
_LENGTH_FACTOR = {"km": 1000.0, "mile": 1609.344}  # 基准：米
_MASS_FACTOR = {"kg": 1.0, "lb": 0.45359237}  # 基准：千克


def _convert_length(value: float, from_unit: str, to_unit: str) -> float:
    return value * _LENGTH_FACTOR[from_unit] / _LENGTH_FACTOR[to_unit]


def _convert_mass(value: float, from_unit: str, to_unit: str) -> float:
    return value * _MASS_FACTOR[from_unit] / _MASS_FACTOR[to_unit]


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    if from_unit == "c":
        return value * 9.0 / 5.0 + 32.0
    return (value - 32.0) * 5.0 / 9.0


# 同组内的四个方向（km<->mile 等）都在表里；跨组与同单位组合不在表里，报错
_CONVERSIONS: dict[tuple[str, str], tuple[str, callable]] = {
    ("km", "mile"): ("length", _convert_length),
    ("mile", "km"): ("length", _convert_length),
    ("kg", "lb"): ("mass", _convert_mass),
    ("lb", "kg"): ("mass", _convert_mass),
    ("c", "f"): ("temperature", _convert_temperature),
    ("f", "c"): ("temperature", _convert_temperature),
}


@mcp.tool()
def text_stats(text: str) -> str:
    """统计文本的字符数、词数（空白切分）、行数。"""
    return json.dumps(
        {
            "characters": len(text),
            "words": len(text.split()),
            "lines": len(text.splitlines()),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算，支持 km/mile、kg/lb、c/f 三组；表外单位组合报错。"""
    key = (from_unit.strip().lower(), to_unit.strip().lower())
    entry = _CONVERSIONS.get(key)
    if entry is None:
        raise ValueError(f"不支持的单位换算: {from_unit} -> {to_unit}（支持 km/mile、kg/lb、c/f）")
    _, converter = entry
    return json.dumps(
        {
            "value": converter(value, key[0], key[1]),
            "from_unit": key[0],
            "to_unit": key[1],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def now_tz(tz_name: str) -> str:
    """返回指定 IANA 时区的当前时间（ISO 8601），如 Asia/Shanghai、UTC。"""
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise ValueError(f"未知时区: {tz_name}") from None
    return json.dumps({"timezone": tz_name, "now": datetime.now(tz).isoformat()}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
