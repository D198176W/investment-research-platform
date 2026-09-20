"""记忆系统 —— 三层记忆架构 + Embedding 向量检索 + 上下文窗口管理

对齐 JD「长期记忆」「多轮上下文管理」「记忆窗口截断、Token 优化」「RAG/Embedding/向量数据库」：

三层记忆：
- 工作记忆（Working）  : 当前任务的中间状态（LangGraph state + scratchpad），由工作流维护
- 短期记忆（Short-term）: 本会话内多轮上下文，由 ContextWindow 做 token 预算与截断
- 长期记忆（Long-term） : 历史任务结论 + Reflector 经验，向量检索注入

模块：
- embeddings.py   : Embedding 提供方（DashScope text-embedding-v3，失败降级为哈希向量）
- vector_store.py : 轻量向量库（numpy + JSON 持久化，无外部依赖）
- memory.py       : LongTermMemory（写入/检索/淘汰）+ MemoryItem
- context.py      : ContextWindow（token 计数、优先级截断、摘要压缩钩子）
"""
