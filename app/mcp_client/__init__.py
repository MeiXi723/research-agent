"""MCP 客户端适配层公开 API。

消费方（app/main.py lifespan）用法：
    from app.mcp_client.provider import MCPProviderManager, parse_servers_spec
    manager = MCPProviderManager(parse_servers_spec(config.MCP_SERVERS))
    await manager.start_all()
    tools = manager.get_tools()          # dict[str, BaseTool]，键 {server}__{tool}
    ...
    await manager.shutdown_all()
"""

from app.mcp_client.provider import (
    NAMESPACE_SEP,
    MCPProviderManager,
    MCPTool,
    parse_servers_spec,
)

__all__ = ["NAMESPACE_SEP", "MCPProviderManager", "MCPTool", "parse_servers_spec"]
