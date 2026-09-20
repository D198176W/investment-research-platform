# 智能投研平台 - Agentic AI 投资研究系统

基于 **LangGraph 状态机 + Function Calling + MCP + 三层记忆 + Reflector 自进化 + 评测驱动** 的 Agent 系统。

> v2 升级：从"固定五阶段流水线"升级为"具备自主规划、工具使用、长期记忆、反思自进化能力的智能体系统"。
> 对齐 2027 秋招大厂 AI 应用开发岗能力要求（字节/小红书/KTC 等）。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│              前端 (Streamlit + 生成式 UI 卡片)                │
│                  http://localhost:8501                       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST + SSE 流式
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway (FastAPI)                       │
│                  http://localhost:8000                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Agent Service (LangGraph 状态机)            │  │
│  │  感知(Agentic) → 建模 → 推理 → 决策 → 反思 → 报告      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ 工具体系  │ │ 记忆系统  │ │ 反思自进化│ │  评测与轨迹    │  │
│  │ MCP+FC   │ │ 三层记忆  │ │ Reflector│ │  Evals+Trace  │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 核心能力（v2 升级）

| 能力 | 实现 | 对齐 JD 要求 |
|------|------|------|
| **LangGraph 图驱动** | 编译后的 StateGraph 真实执行（graph.stream），条件入口支持断点续跑 | Agent Harness / Runtime 工程 |
| **工具协议化** | 15 个市场数据工具，JSON Schema 描述，模型动态选择 | Function Calling、Tool Use |
| **MCP 体系** | MCP Server (stdio) 暴露工具，Client 动态发现调用；主链路可配置经 MCP 执行 | MCP 工具与 Skills 体系 |
| **Skills 体系** | 价值投资/事件驱动/产业链 3 个方法论 Skill，按需注入 | Skills 体系 |
| **三层记忆** | 工作记忆(state)/短期记忆(上下文窗口)/长期记忆(向量库) | Memory/Context Management |
| **上下文管理** | tiktoken 精确计数，工具循环内协议安全裁剪（系统>任务>近期证据） | 记忆窗口截断、Token 优化 |
| **RAG 检索** | DashScope Embedding + 向量库，历史结论语义召回 | RAG、Embedding、向量数据库 |
| **Agentic Search** | 感知阶段 ReAct 工具调用循环，模型自主规划收集数据 | Agentic Search、自主规划 |
| **反思自进化** | Reflector 独立审核模型复核决策，修正意见回灌推理 Prompt，经验写回记忆 | Reflector 自进化、AI 自我进化 |
| **状态机回滚** | 每阶段独立重试(3次)，失败回退上一阶段，条件边全部由图承担 | 状态管理及异常恢复 |
| **评测驱动** | 20 条评测集 + 9 项规则校验 + LLM-as-Judge + 56 项单测 | 评测驱动研发体系 |
| **全链路 Trace** | 每步 LLM/Tool 调用记录（真实 token 用量 + tiktoken 估算），落盘可回放 | 推理轨迹构建、可观测性 |
| **多模型路由** | ModelRouter 降级链接入主链路，403 配额耗尽自动切换；审核官独立模型 | 多模型、部署与微调 |
| **流式响应** | SSE 实时推送任务进度，前端订阅动态渲染阶段时间线 | 流式、生成式 UI |
| **安全加固** | API 令牌桶限流、CORS 白名单、工具只读护栏、参数校验 | 安全 |
| **容器化部署** | Dockerfile + Docker Compose 一键部署 + GitHub Actions CI | 云服务平台、容器化部署 |

## 功能特性

- **六阶段 Agent 流程**：感知(Agentic) → 建模 → 推理 → 决策 → 反思 → 报告，由 LangGraph 编译图真实驱动
- **自主工具调用**：模型基于工具描述自主选择工具、生成参数、解析结果、失败恢复
- **真实数据校验**：方案假设与真实行情数据比对，自动调整置信度（矛盾惩罚/吻合奖励）
- **多模型交叉评估**：qwen-max + qwen-plus 独立评分融合，减少单一模型偏差
- **分析-审核双角色**：Reflector 使用独立审核模型（`REVIEWER_MODEL`）复核决策，修正意见真实回灌推理 Prompt
- **多模型降级链**：主模型 403 配额耗尽时 ModelRouter 自动切换备用模型，任务不中断
- **断点续跑**：每阶段检查点持久化，服务重启后经图条件入口从中断阶段直接恢复
- **历史记录管理**：保存、查询、删除历史分析记录
- **PDF 报告导出**：生成专业格式的 PDF 投资报告

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DashScope API 密钥
```

### 3. 启动服务

#### 方式一：启动脚本 (Windows)
```bash
start.bat
```

#### 方式二：手动启动

**后端：**
```bash
cd backend/gateway
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**前端：**
```bash
cd frontend
streamlit run streamlit_app.py --server.port 8501
```

#### 方式三：Docker Compose（推荐）
```bash
cp .env.example .env   # 填入 DASHSCOPE_API_KEY
docker compose up -d --build
```

### 4. 访问服务

- 前端界面：http://localhost:8501
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 项目结构

```
investment-research-platform/
├── backend/
│   ├── gateway/              # FastAPI API 网关
│   │   ├── main.py           # 主应用（含 SSE 流式、Agent 能力观测端点）
│   │   ├── config.py         # 配置管理
│   │   └── models.py         # Pydantic 数据模型
│   ├── agent_service/        # Agent 服务（核心）
│   │   ├── workflow.py       # LangGraph 六阶段状态机工作流
│   │   ├── agent_loop.py     # ReAct 工具调用循环（Agentic 感知核心）
│   │   ├── llm_client.py     # OpenAI 兼容客户端 + 多模型路由/降级
│   │   ├── reflector.py      # Reflector 反思审核节点
│   │   ├── skills.py         # Skills 加载与选择
│   │   ├── tracer.py         # 全链路执行轨迹（Trace）
│   │   ├── validation.py     # Pydantic 校验工具
│   │   ├── tools/            # 工具体系
│   │   │   ├── registry.py   #   Tool 注册中心（JSON Schema）
│   │   │   ├── market_tools.py#  15 个市场数据工具
│   │   │   ├── executor.py   #   工具执行器（异常分类+重试+截断）
│   │   │   ├── mcp_server.py #   MCP Server（stdio）
│   │   │   └── mcp_client.py #   MCP Client（动态工具发现）
│   │   └── memory/           # 记忆系统
│   │       ├── embeddings.py #   Embedding（DashScope + 哈希降级）
│   │       ├── vector_store.py#  轻量向量库（numpy + JSON）
│   │       ├── memory.py     #   长期记忆（写入/检索/淘汰）
│   │       └── context.py    #   上下文窗口管理（Token 截断）
│   ├── data_fetcher/         # AKShare 数据采集（被工具封装）
│   ├── data_service/         # 历史记录管理
│   └── report_service/       # PDF 报告生成
├── skills/                   # Skills 方法论（Markdown）
│   ├── value_investing.md    #   价值投资分析框架
│   ├── event_driven.md       #   事件驱动分析框架
│   └── industry_chain.md     #   产业链分析框架
├── evals/                    # 评测体系
│   ├── dataset/              #   评测集（20 条带期望要点的任务）
│   ├── metrics.py            #   规则校验指标（9 项：完成率/方向覆盖/数据校验生效等）
│   └── runner.py             #   评测 Runner（含 LLM-as-Judge）
├── tests/                    # 单元测试（56 项：工具/执行器/上下文/向量库/工作流/路由/评测）
├── frontend/
│   └── streamlit_app.py      # Streamlit 前端（SSE 流式 + 生成式 UI 卡片 + Agent 观测台）
├── Dockerfile.backend        # 后端容器镜像
├── Dockerfile.frontend       # 前端容器镜像
├── docker-compose.yml        # 一键容器化部署
└── .github/workflows/ci.yml  # CI（lint + 单测 + 评测冒烟）
```

## API 接口

### 研究分析
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/research` | 创建研究任务（支持 `use_agentic_perception` 开关） |
| GET | `/api/v1/research` | 列出研究任务 |
| GET | `/api/v1/research/{task_id}` | 获取任务状态和结果 |
| GET | `/api/v1/research/{task_id}/stream` | **SSE 流式推送进度** |
| GET | `/api/v1/research/{task_id}/trace` | **获取执行轨迹（LLM/Tool 明细）** |
| POST | `/api/v1/research/{task_id}/export-pdf` | 导出 PDF 报告 |

### Agent 能力观测（v2 新增）
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/agent/tools` | 列出全部工具（动态发现） |
| GET | `/api/v1/agent/mcp/status` | MCP Server 探活与工具发现 |
| GET | `/api/v1/agent/memory/stats` | 长期记忆统计 |
| GET | `/api/v1/agent/tools/metrics` | 工具级指标（成功率/时延/异常分布） |

### 市场数据 / 历史 / 配置
| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/v1/market/*` | 行情/指数/板块/新闻/财务等 |
| GET | `/api/v1/history` | 历史记录 |
| GET | `/api/v1/config` | 当前配置 |

## 评测

```bash
cd backend
# 运行评测集（规则校验 + LLM-as-Judge）
python -m evals.runner

# 只跑 1 条（调试）
python -m evals.runner --max-tasks 1

# 禁用 LLM-as-Judge（省 token，只做规则校验）
python -m evals.runner --no-judge

# 单元测试（56 项，不依赖 LLM 与网络）
cd .. && python -m pytest tests/ -q
```

评测报告输出到 `evals/reports/eval_report_{timestamp}.json`。

## 新增一个工具（面试演示点）

只需两步，Agent 与 MCP Server 自动可见，无需改其他代码：

```python
# backend/agent_service/tools/market_tools.py
@register_tool(
    name="get_new_data",
    description="获取某类数据",
    parameters={
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "股票代码"}},
        "required": ["symbol"],
    },
)
def tool_get_new_data(symbol: str):
    stock, *_ = _lazy_fetchers()
    return _guard(stock.some_method(symbol), f"新数据({symbol})")
```

## 技术栈

- **Agent 框架**：LangGraph（状态机）、LangChain
- **LLM**：DashScope（通义千问），OpenAI 兼容协议
- **工具协议**：Function Calling、MCP（Model Context Protocol）
- **记忆**：numpy 向量库 + DashScope Embedding + tiktoken
- **后端**：FastAPI, Uvicorn, Pydantic
- **前端**：Streamlit（生成式 UI 卡片）
- **数据源**：AKShare
- **部署**：Docker, Docker Compose, GitHub Actions CI

## 开发说明

### 添加新的工作流阶段
在 `backend/agent_service/workflow.py` 中添加节点函数，更新 `PHASE_ORDER` 与 `get_conditional_mapping()`。

### 添加新的 Skill
在 `skills/` 目录新增 Markdown 文件（含 YAML 头部元数据），Agent 自动加载。

### 运行 MCP Server 独立调试
```bash
cd backend
python -m agent_service.tools.mcp_server
```

## 许可证

MIT License
