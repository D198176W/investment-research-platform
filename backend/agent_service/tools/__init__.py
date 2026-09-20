"""Agent 工具体系 —— Tool 注册中心 + MCP Server/Client + Function Calling

模块结构：
- registry.py   : Tool 定义与注册中心（JSON Schema 描述，模型可动态选择）
- executor.py   : 工具执行器（结果理解：校验/截断/摘要，异常分类与失败恢复）
- mcp_server.py : 将注册中心中的工具暴露为 MCP Server（stdio 传输）
- mcp_client.py : MCP Client，启动时动态拉取工具清单（tools/list）、调用（tools/call）
"""
