"""流水线配置加载与校验（pipelines/*.json，配置驱动的多智能体编排）。

roles 顺序即链路执行顺序；route 描述某节点的条件回跳。
路径安全：流水线名只允许 [a-z0-9_-]+，在拼路径之前就拒绝（../、盘符、
反斜杠、空白等一切穿越/变体形态都不可能通过白名单），不做 resolve 后比对。
"""

import json
import re
from pathlib import Path

# 名字白名单：非空的小写字母/数字/下划线/连字符
PIPELINE_NAME_RE = re.compile(r"^[a-z0-9_-]+$")
VALID_ENGINES = ("research", "chain")
# route.max_loops 缺省值：单个节点最多回跳几次
DEFAULT_MAX_LOOPS = 2

# 仓库根下的 pipelines/ 目录（本文件位于 app/multi_agent/，parents[2] 即仓库根）
PIPELINES_DIR = Path(__file__).resolve().parents[2] / "pipelines"


def validate_pipeline_config(config: dict) -> None:
    """校验流水线配置；失败抛 ValueError，消息带流水线/角色名便于定位。

    route 的校验放在角色名收齐之后：revise_to 允许指向清单中任意角色
    （包括定义在它后面的），所以不能在逐角色扫描时顺带判存在性。
    """
    if not isinstance(config, dict):
        raise ValueError("流水线配置必须是 JSON 对象")
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("流水线配置缺少非空字符串字段 name")
    engine = config.get("engine")
    if engine not in VALID_ENGINES:
        raise ValueError(
            f"流水线 {name}: engine 必须是 {'/'.join(VALID_ENGINES)} 之一，收到 {engine!r}"
        )
    if not isinstance(config.get("description", ""), str):
        raise ValueError(f"流水线 {name}: description 必须是字符串")
    if "default" in config and not isinstance(config["default"], bool):
        raise ValueError(f"流水线 {name}: default 必须是布尔值")

    roles = config.get("roles", [])
    if not isinstance(roles, list):
        raise ValueError(f"流水线 {name}: roles 必须是数组")
    if engine == "chain" and not roles:
        raise ValueError(f"流水线 {name}: chain 引擎至少要有一个角色（roles 为空）")

    role_names: list[str] = []
    for idx, role in enumerate(roles):
        _validate_role_basic(name, idx, role, role_names)
    for role in roles:
        if role.get("route") is not None:
            _validate_route(name, role, role_names)


def _validate_role_basic(pline_name: str, idx: int, role, role_names: list[str]) -> None:
    if not isinstance(role, dict):
        raise ValueError(f"流水线 {pline_name}: roles[{idx}] 必须是对象")
    role_name = role.get("name")
    if not isinstance(role_name, str) or not role_name.strip():
        raise ValueError(f"流水线 {pline_name}: roles[{idx}] 缺少非空字符串 name")
    if role_name in role_names:
        raise ValueError(f"流水线 {pline_name}: 角色名重复: {role_name}")
    role_names.append(role_name)
    prompt = role.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"角色 {role_name}: 缺少非空字符串 system_prompt")
    for field in ("tools", "output_fields"):
        value = role.get(field, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"角色 {role_name}: {field} 必须是字符串数组")


def _validate_route(pline_name: str, role: dict, role_names: list[str]) -> None:
    prefix = f"流水线 {pline_name} 角色 {role['name']}"
    route = role.get("route")
    if not isinstance(route, dict):
        raise ValueError(f"{prefix}: route 必须是对象")
    pass_field = route.get("pass_field")
    if not isinstance(pass_field, str) or not pass_field:
        raise ValueError(f"{prefix}: route.pass_field 必须是非空字符串")
    output_fields = role.get("output_fields", [])
    if pass_field not in output_fields:
        raise ValueError(
            f"{prefix}: route.pass_field {pass_field!r} 不在 output_fields {output_fields} 里"
        )
    if "pass_value" not in route:
        raise ValueError(f"{prefix}: route 缺少 pass_value")
    revise_to = route.get("revise_to")
    if not isinstance(revise_to, str) or not revise_to:
        raise ValueError(f"{prefix}: route.revise_to 必须是非空字符串")
    if revise_to not in role_names:
        raise ValueError(f"{prefix}: revise_to {revise_to!r} 不在角色清单 {role_names} 里")
    max_loops = route.get("max_loops", DEFAULT_MAX_LOOPS)
    # bool 是 int 子类，需先排除，避免 True 被当成 1
    if isinstance(max_loops, bool) or not isinstance(max_loops, int) or max_loops < 1:
        raise ValueError(f"{prefix}: route.max_loops 必须是 >=1 的整数")


def load_pipeline(name: str) -> dict:
    """按名加载 pipelines/<name>.json 并校验。

    非法名（含路径穿越形态）抛 ValueError；文件不存在抛 FileNotFoundError。
    """
    if not isinstance(name, str) or not PIPELINE_NAME_RE.match(name):
        raise ValueError(f"非法流水线名: {name!r}（只允许 [a-z0-9_-]+）")
    path = PIPELINES_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"流水线配置不存在: {name}（查找路径 {path}）")
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"流水线 {name}: JSON 解析失败: {e}") from e
    validate_pipeline_config(config)
    return config


def list_pipelines() -> list[dict]:
    """扫描 pipelines/ 目录，返回每个配置的元信息（按文件名排序）。

    单个文件损坏不影响其它条目：该文件以 {"name", "error"} 形式呈现。
    """
    items: list[dict] = []
    if not PIPELINES_DIR.is_dir():
        return items
    for path in sorted(PIPELINES_DIR.glob("*.json")):
        stem = path.stem
        # 文件名本身也过白名单：目录被塞入奇怪文件时不会泄出
        if not PIPELINE_NAME_RE.match(stem):
            continue
        try:
            config = load_pipeline(stem)
        except (ValueError, OSError) as e:
            items.append({"name": stem, "error": str(e)})
            continue
        items.append({
            "name": config.get("name", stem),
            "description": config.get("description", ""),
            "engine": config.get("engine"),
            "default": bool(config.get("default", False)),
            "roles": [r.get("name") for r in config.get("roles", [])],
        })
    return items
