"""智能投研平台 - Streamlit 前端 v2

设计目标：
- 现代金融科技视觉：深色主题 + 玻璃拟态卡片 + 渐变点缀
- SSE 流式进度：实时订阅后端执行状态，动态渲染阶段时间线
- 生成式 UI：按 Agent 执行结果动态生成卡片（方案评分条、反思徽章、工具轨迹）
- Agent 观测台：工具清单 / MCP 状态 / 长期记忆 / 工具指标 / 执行轨迹回放
"""
import json
import os
import time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="智能投研平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ==================== 全局样式 ====================

st.markdown("""
<style>
:root {
    --bg-card: #FFFFFF;
    --bg-card-hover: #F7F9FD;
    --border: #E5EAF2;
    --accent: #2F6BFF;
    --accent2: #7C5CFF;
    --green: #16A34A;
    --red: #DC2626;
    --amber: #D97706;
    --text-dim: rgba(26,34,51,0.58);
}
/* 页面背景（白色基调 + 淡彩光晕） */
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(47,107,255,0.07), transparent 60%),
                     radial-gradient(900px 500px at -10% 110%, rgba(124,92,255,0.06), transparent 55%),
                     #F6F8FC; color: #1A2233; }
/* 顶栏 */
.app-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
    background: linear-gradient(120deg, #2F6BFF, #7C5CFF);
    border-radius: 18px; box-shadow: 0 8px 24px rgba(47,107,255,0.25);
}
.app-header .title { font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px;
    color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; }
.app-header .subtitle { color: rgba(255,255,255,0.85); font-size: 0.85rem; margin-top: 0.25rem; }
.status-pill { display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.35rem 0.9rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    background: rgba(255,255,255,0.18); color: #FFFFFF; border: 1px solid rgba(255,255,255,0.4); }
.status-pill.bad { background: rgba(220,38,38,0.12); color: #DC2626; border-color: rgba(220,38,38,0.4); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.dot.pulse { animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* 指数行情条 */
.ticker-card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px;
    padding: 0.9rem 1.1rem; text-align: left; transition: all .2s ease;
    box-shadow: 0 1px 3px rgba(26,34,51,0.05);
}
.ticker-card:hover { background: var(--bg-card-hover); transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(26,34,51,0.08); }
.ticker-name { font-size: 0.78rem; color: var(--text-dim); font-weight: 600; }
.ticker-price { font-size: 1.25rem; font-weight: 800; margin: 0.15rem 0; font-variant-numeric: tabular-nums; }
.ticker-chg { font-size: 0.8rem; font-weight: 700; }
.up { color: #DC2626; }      /* A股红涨 */
.down { color: #059669; }    /* A股绿跌 */
.flat { color: var(--text-dim); }

/* 通用卡片 */
.card {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.2rem 1.4rem; margin: 0.7rem 0;
    box-shadow: 0 1px 3px rgba(26,34,51,0.05);
}
.card-title { font-size: 0.95rem; font-weight: 700; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.5rem; color: #1A2233; }
.card-title .bar { width: 4px; height: 16px; border-radius: 2px;
    background: linear-gradient(180deg, var(--accent), var(--accent2)); }

/* 阶段时间线 */
.phase-flow { display: flex; gap: 6px; margin: 1rem 0; }
.phase-node { flex: 1; text-align: center; padding: 0.55rem 0.2rem; border-radius: 10px;
    font-size: 0.78rem; font-weight: 700; border: 1px solid var(--border);
    background: var(--bg-card); color: var(--text-dim); transition: all .3s ease; }
.phase-node.done { background: #E7F8EF; color: #15803D; border-color: #B8E6CC; }
.phase-node.active { background: #EAF1FF; color: #2563EB;
    border-color: #B9CEFF; animation: glow 1.6s ease-in-out infinite; }
@keyframes glow { 0%,100% { box-shadow: 0 0 0 0 rgba(47,107,255,0.0); }
                  50% { box-shadow: 0 0 14px 1px rgba(47,107,255,0.30); } }

/* 评分条 */
.score-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.35rem 0; font-size: 0.82rem; }
.score-label { width: 5.5rem; color: var(--text-dim); flex-shrink: 0; }
.score-track { flex: 1; height: 8px; border-radius: 4px; background: #EEF2F8; overflow: hidden; }
.score-fill { height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.score-val { width: 3rem; text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }

/* 标签/徽章 */
.tag { display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px; font-size: 0.74rem;
    font-weight: 600; margin: 0.15rem 0.2rem 0.15rem 0; border: 1px solid #CBDBFF;
    background: #EAF1FF; color: #2563EB; }
.tag.green { background: #E7F8EF; color: #15803D; border-color: #B8E6CC; }
.tag.red { background: #FDECEC; color: #DC2626; border-color: #F5C2C2; }
.tag.amber { background: #FEF4E0; color: #B45309; border-color: #F5DBB0; }
.tag.purple { background: #F1EBFF; color: #6D28D9; border-color: #DCCBFF; }

/* 指标格 */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.6rem; }
.metric-cell { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px;
    padding: 0.75rem 0.9rem; box-shadow: 0 1px 3px rgba(26,34,51,0.05); }
.metric-cell .k { font-size: 0.74rem; color: var(--text-dim); }
.metric-cell .v { font-size: 1.05rem; font-weight: 800; margin-top: 0.2rem;
    font-variant-numeric: tabular-nums; word-break: break-all; color: #1A2233; }

/* 报告容器 */
.report-box { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.6rem 1.8rem; line-height: 1.85; white-space: pre-wrap;
    max-height: 640px; overflow-y: auto; font-size: 0.92rem; color: #1A2233;
    box-shadow: 0 1px 3px rgba(26,34,51,0.05); }

/* Trace 时间线 */
.trace-step { border-left: 2px solid rgba(47,107,255,0.45); padding: 0.4rem 0 0.4rem 1rem;
    margin-left: 0.4rem; position: relative; }
.trace-step::before { content: ""; position: absolute; left: -6px; top: 0.9rem;
    width: 10px; height: 10px; border-radius: 50%; background: var(--accent); }
.trace-step.error { border-left-color: rgba(220,38,38,0.5); }
.trace-step.error::before { background: var(--red); }

/* 表格 */
table.obs { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
table.obs th { text-align: left; color: var(--text-dim); font-weight: 600;
    padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); }
table.obs td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #F0F3F8; color: #1A2233; }
table.obs code { background: #EEF2F8; padding: 0.1rem 0.35rem; border-radius: 5px; }

/* Streamlit 组件微调 */
div[data-testid="stTabs"] button { font-weight: 700; }
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)); }
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: #1A2233; }
section[data-testid="stSidebar"] hr { border-color: #E5EAF2; }
/* 内嵌指标卡（决策/个股速查等内联卡片文字） */
.card b, .card div { color: inherit; }
.card { color: #1A2233; }
/* 结论高亮块（决策建议） */
.card div[style*="background:rgba(79,140,255,0.1)"] { color: #1A2233; }
</style>
""", unsafe_allow_html=True)

# ==================== Session State ====================

_DEFAULTS = {
    "model_name": "qwen-plus",
    "current_task_id": None,
    "current_analysis": None,
    "analysis_complete": False,
    "pdf_data": None,
    "market_indices": None,
    "market_news": None,
    "indices_last_update": None,
    "news_last_update": None,
    "live_phase": None,
    "live_progress": 0,
    "live_events": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== API 工具函数 ====================

def api_get(endpoint: str, params: dict = None, timeout: int = 30) -> dict:
    try:
        resp = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "无法连接后端服务"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_post(endpoint: str, data: dict = None, timeout: int = 30) -> dict:
    try:
        resp = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            body = e.response.json()
            detail = body.get("detail") or body.get("message") or str(body)
        except Exception:
            detail = str(e)
        return {"success": False, "error": f"HTTP {e.response.status_code}: {detail}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def api_delete(endpoint: str) -> dict:
    try:
        resp = requests.delete(f"{API_BASE_URL}{endpoint}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_api_health() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=4).status_code == 200
    except Exception:
        return False


def stream_task_events(task_id: str):
    """订阅 SSE 流，逐条产出事件 dict（生成器）"""
    with requests.get(f"{API_BASE_URL}/api/v1/research/{task_id}/stream",
                      stream=True, timeout=(5, 900)) as resp:
        resp.raise_for_status()
        event_name = "message"
        data_buf = []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_buf.append(line[5:].strip())
            elif line == "":
                if data_buf:
                    try:
                        yield event_name, json.loads("".join(data_buf))
                    except json.JSONDecodeError:
                        pass
                    data_buf = []
                    event_name = "message"
                if event_name == "done":
                    break


# ==================== 数据加载 ====================

def load_market_indices():
    now = datetime.now()
    if st.session_state.market_indices and st.session_state.indices_last_update:
        if (now - st.session_state.indices_last_update).total_seconds() < 60:
            return st.session_state.market_indices
    result = api_get("/api/v1/market/indices")
    if result.get("success"):
        st.session_state.market_indices = result.get("data", [])
        st.session_state.indices_last_update = now
        return st.session_state.market_indices
    return None


def load_market_news():
    now = datetime.now()
    if st.session_state.market_news and st.session_state.news_last_update:
        if (now - st.session_state.news_last_update).total_seconds() < 300:
            return st.session_state.market_news
    result = api_get("/api/v1/market/news", params={"keyword": "财经", "limit": 6})
    if result.get("success"):
        st.session_state.market_news = result.get("data", [])
        st.session_state.news_last_update = now
        return st.session_state.market_news
    return None


def load_history():
    result = api_get("/api/v1/history")
    return result.get("data", []) if result.get("success") else []


# ==================== 渲染组件 ====================

PHASES = [
    ("perception", "感知"), ("modeling", "建模"), ("reasoning", "推理"),
    ("decision", "决策"), ("reflect", "反思"), ("report", "报告"),
]
PHASE_NAME = dict(PHASES)


def render_phase_flow(current_phase: str, status: str = "running"):
    """阶段时间线（done / active / pending 三态）"""
    keys = [k for k, _ in PHASES]
    if status == "completed":
        active_idx = len(keys)
    elif current_phase in keys:
        active_idx = keys.index(current_phase)
    else:
        active_idx = 0
    html = '<div class="phase-flow">'
    for i, (key, name) in enumerate(PHASES):
        style = ""
        if status == "failed" and i == active_idx:
            cls, suffix = "phase-node", " ✕"
            style = (' style="background:#FDECEC;color:#DC2626;'
                     'border-color:#F5C2C2;"')
        elif i < active_idx or status == "completed":
            cls, suffix = "phase-node done", " ✓"
        elif i == active_idx:
            cls, suffix = "phase-node active", ""
        else:
            cls, suffix = "phase-node", ""
        html += f'<div class="{cls}"{style}>{name}{suffix}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def score_bar(label: str, value: float, weight: str = "") -> str:
    pct = max(0.0, min(1.0, float(value or 0))) * 100
    w = f'<span style="color:var(--text-dim);font-size:0.72rem;">{weight}</span>' if weight else ""
    return (f'<div class="score-row"><span class="score-label">{label}{w}</span>'
            f'<div class="score-track"><div class="score-fill" style="width:{pct:.0f}%"></div></div>'
            f'<span class="score-val">{pct:.0f}%</span></div>')


def render_ticker():
    indices = load_market_indices()
    st.markdown('<div class="card-title" style="margin:0.2rem 0 0.6rem;">'
                '<span class="bar"></span>实时市场概览</div>', unsafe_allow_html=True)
    if not indices:
        st.caption("行情数据加载中（后端 AKShare 数据源）...")
        return
    items = indices if isinstance(indices, list) else list(indices.values())
    items = [i for i in items if isinstance(i, dict) and "error" not in i][:6]
    cols = st.columns(max(len(items), 1))
    for col, idx in zip(cols, items):
        with col:
            name = idx.get("name", "--")
            price = idx.get("price", "--")
            try:
                chg = float(str(idx.get("change_pct", 0)).replace("%", "").replace("+", ""))
            except (ValueError, TypeError):
                chg = 0.0
            cls = "up" if chg > 0 else ("down" if chg < 0 else "flat")
            arrow = "▲" if chg > 0 else ("▼" if chg < 0 else "—")
            price_txt = f"{price:,.2f}" if isinstance(price, (int, float)) else price
            st.markdown(
                f'<div class="ticker-card"><div class="ticker-name">{name}</div>'
                f'<div class="ticker-price">{price_txt}</div>'
                f'<div class="ticker-chg {cls}">{arrow} {chg:+.2f}%</div></div>',
                unsafe_allow_html=True)
    if st.session_state.indices_last_update:
        st.caption(f"更新于 {st.session_state.indices_last_update.strftime('%H:%M:%S')} · 数据源 AKShare")


# ==================== 侧边栏 ====================

api_healthy = check_api_health()

with st.sidebar:
    st.markdown("### 引擎配置")

    QWEN_MODELS = ["qwen-plus", "qwen-max", "qwen-turbo", "qwen3.8-27b",
                   "qwen3.7-max", "qwen3.7-plus"]
    idx = QWEN_MODELS.index(st.session_state.model_name) \
        if st.session_state.model_name in QWEN_MODELS else 0
    st.session_state.model_name = st.selectbox(
        "主分析模型", options=QWEN_MODELS, index=idx,
        help="重推理任务模型；配额耗尽(403)时自动沿降级链切换 qwen-plus")
    st.caption("平台：阿里云 DashScope · 密钥由后端 .env 管理")

    st.divider()
    st.markdown("### 研究任务")

    TOPIC_INDUSTRY_MAP = {
        "人工智能芯片市场": ["半导体", "人工智能"],
        "新能源汽车产业链": ["汽车", "新能源", "新材料"],
        "创新药研发管线": ["生物医药"],
        "跨境电商出海机遇": ["互联网", "消费电子"],
        "低空经济产业链": ["航空航天", "智能制造", "新能源"],
        "储能技术商业化": ["新能源", "新材料", "智能制造"],
        "人形机器人产业": ["智能制造", "人工智能", "新材料"],
        "消费级AR/VR市场": ["消费电子", "传媒娱乐", "人工智能"],
        "半导体设备国产化": ["半导体", "新材料", "智能制造"],
        "金融科技监管趋势": ["金融科技", "互联网"],
        "碳中和绿色投资": ["新能源", "新材料", "智能制造"],
        "量子计算商用化": ["半导体", "人工智能", "新材料"],
    }
    ALL_INDUSTRIES = ["半导体", "人工智能", "新能源", "生物医药", "消费电子",
                      "金融科技", "智能制造", "航空航天", "新材料", "互联网",
                      "汽车", "传媒娱乐"]

    topic_option = st.selectbox("研究主题", options=["自定义输入"] + list(TOPIC_INDUSTRY_MAP.keys()), index=1)
    if topic_option == "自定义输入":
        research_topic = st.text_input("输入研究主题", value="")
        available_industries = ALL_INDUSTRIES
    else:
        research_topic = topic_option
        available_industries = TOPIC_INDUSTRY_MAP.get(topic_option, ALL_INDUSTRIES)

    industry_option = st.selectbox("行业焦点", options=["自定义输入"] + available_industries)
    industry_focus = st.text_input("输入行业焦点") if industry_option == "自定义输入" else industry_option

    time_horizon = st.selectbox("时间范围",
                                ["短期(1-3个月)", "中期(3-6个月)", "中期(6-12个月)", "长期(1年以上)"], index=2)
    start_analysis = st.button("启动智能分析", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### 个股速查")
    symbol_input = st.text_input("A 股代码", placeholder="如 600519", label_visibility="collapsed")
    if st.button("查询行情", use_container_width=True) and symbol_input.strip():
        with st.spinner("查询中..."):
            q_res = api_get(f"/api/v1/market/quote/{symbol_input.strip()}")
        if q_res.get("success"):
            q = q_res.get("data", {})
            chg = q.get("change_pct", 0) or 0
            cls = "up" if chg >= 0 else "down"
            arrow = "▲" if chg >= 0 else "▼"
            st.markdown(
                f'<div class="card" style="margin:0.3rem 0;">'
                f'<div style="font-weight:800;">{q.get("name", "")} <span style="color:var(--text-dim);'
                f'font-size:0.8rem;">{q.get("symbol", symbol_input)}</span></div>'
                f'<div style="display:flex;justify-content:space-between;margin-top:0.4rem;">'
                f'<span style="font-size:1.3rem;font-weight:800;">{q.get("price", "--")}</span>'
                f'<span class="{cls}" style="font-weight:700;">{arrow} {chg}%</span></div>'
                f'<div style="color:var(--text-dim);font-size:0.74rem;margin-top:0.4rem;">'
                f'今开 {q.get("open", "--")} · 最高 {q.get("high", "--")} · 最低 {q.get("low", "--")}<br>'
                f'PE {q.get("pe_ratio", "--")} · PB {q.get("pb_ratio", "--")} · {q.get("data_source", "")}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.error(q_res.get("error", "查询失败"))

    st.divider()
    st.markdown("### 财经快讯")
    news = load_market_news()
    if news:
        for n in news[:5]:
            title = n.get("title", "")
            short = title[:24] + "…" if len(title) > 24 else title
            with st.expander(short, expanded=False):
                st.caption(f"{n.get('source', '')} · {n.get('publish_time', '')}")
                content = (n.get("content") or "").strip()
                st.markdown(content[:400] if content else "暂无摘要")
                if n.get("url"):
                    st.markdown(f"[阅读原文]({n['url']})")
    else:
        st.caption("快讯加载中...")

# ==================== 顶栏 ====================

health_pill = ('<span class="status-pill ok"><span class="dot pulse"></span>引擎在线</span>'
               if api_healthy else
               '<span class="status-pill bad"><span class="dot"></span>引擎离线</span>')
st.markdown(
    f'<div class="app-header"><div>'
    f'<div class="title">智能投研平台</div>'
    f'<div class="subtitle">LangGraph 状态机 · Function Calling · MCP 工具生态 · 三层记忆 · Reflector 自进化 · 评测驱动</div>'
    f'</div><div style="text-align:right;">{health_pill}'
    f'<div style="color:rgba(255,255,255,0.8);font-size:0.72rem;margin-top:0.4rem;">{API_BASE_URL}</div>'
    f'</div></div>', unsafe_allow_html=True)

render_ticker()
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

# ==================== 主标签页 ====================

tab_work, tab_observe, tab_trace, tab_history = st.tabs(
    ["分析工作台", "Agent 观测台", "执行轨迹", "历史档案"])

# -------------------- 分析工作台 --------------------
with tab_work:
    if start_analysis:
        if not api_healthy:
            st.error("后端服务未启动，请先启动 FastAPI 网关（uvicorn main:app --port 8000）")
        elif not research_topic or not industry_focus:
            st.warning("请填写研究主题与行业焦点")
        else:
            result = api_post("/api/v1/research", {
                "topic": research_topic, "industry": industry_focus,
                "horizon": time_horizon, "model_name": st.session_state.model_name,
            })
            if result.get("success"):
                task_id = result["data"]["task_id"]
                st.session_state.current_task_id = task_id
                st.session_state.analysis_complete = False
                st.session_state.live_events = []
                st.session_state.current_analysis = None

                flow_ph = st.empty()
                prog_ph = st.empty()
                log_ph = st.empty()

                try:
                    for event, payload in stream_task_events(task_id):
                        if event == "done":
                            break
                        phase = payload.get("current_phase", "perception")
                        progress = payload.get("progress_percent", 0)
                        status = payload.get("status", "running")
                        st.session_state.live_phase = phase
                        with flow_ph.container():
                            render_phase_flow(phase, status)
                        prog_ph.progress(min(progress, 100) / 100,
                                         text=f"{PHASE_NAME.get(phase, phase)} · {progress}%")
                        events = st.session_state.live_events
                        if not events or events[-1] != phase:
                            events.append(phase)
                            log_ph.caption(" → ".join(PHASE_NAME.get(p, p) for p in events))
                except Exception as e:
                    st.warning(f"SSE 流中断，转为轮询获取结果（{e}）")

                # 拉取最终结果
                final = api_get(f"/api/v1/research/{task_id}")
                if final.get("success"):
                    st.session_state.current_analysis = final.get("data", {})
                    st.session_state.analysis_complete = (
                        st.session_state.current_analysis.get("status") == "completed")
                st.session_state.pdf_data = None
                st.rerun()
            else:
                st.error(f"任务创建失败：{result.get('error')}")

    # ---------- 结果渲染（生成式 UI） ----------
    analysis = st.session_state.current_analysis
    if analysis:
        status = analysis.get("status", "")
        phase = analysis.get("current_phase", "")
        render_phase_flow(phase, status)

        if status == "failed":
            err = analysis.get("error", "未知错误")
            st.markdown(
                f'<div class="card" style="border-color:rgba(239,68,68,0.4);">'
                f'<div class="card-title"><span class="bar" style="background:var(--red);"></span>'
                f'分析失败</div><div style="font-size:0.88rem;">{err}</div></div>',
                unsafe_allow_html=True)
            with st.expander("排查建议"):
                if "403" in str(err) or "配额" in str(err):
                    st.markdown("- 模型配额耗尽：在侧边栏更换模型，或等待降级链自动切换")
                elif "api key" in str(err).lower() or "认证" in str(err):
                    st.markdown("- 检查项目根目录 .env 的 DASHSCOPE_API_KEY 并重启后端")
                else:
                    st.markdown("- 查看「执行轨迹」标签页定位失败步骤")

        # --- 感知阶段 ---
        p = analysis.get("perception_data")
        if p:
            with st.expander("① 感知 · 市场数据与 Agentic 工具轨迹", expanded=True):
                sources = p.get("_realtime_data_sources", [])
                if sources:
                    st.markdown("".join(f'<span class="tag green">{s}</span>' for s in sources),
                                unsafe_allow_html=True)
                agentic = p.get("_agentic")
                if agentic:
                    st.markdown(
                        f'<div style="margin:0.5rem 0;">'
                        f'<span class="tag purple">Agentic 循环 {agentic.get("rounds")} 轮</span>'
                        f'<span class="tag">工具调用 {agentic.get("tool_calls")} 次</span>'
                        + (f'<span class="tag amber">Skill · {agentic.get("skill")}</span>'
                           if agentic.get("skill") else "")
                        + "</div>", unsafe_allow_html=True)
                st.markdown(f"**市场概况**　{p.get('market_overview', '—')}")
                indicators = p.get("key_indicators", {})
                if indicators:
                    cells = "".join(
                        f'<div class="metric-cell"><div class="k">{k}</div><div class="v">{v}</div></div>'
                        for k, v in list(indicators.items())[:6])
                    st.markdown(f'<div class="metric-grid" style="margin:0.6rem 0;">{cells}</div>',
                                unsafe_allow_html=True)
                for n_item in (p.get("recent_news") or [])[:3]:
                    st.markdown(f"- {n_item}")
                trends = p.get("industry_trends", {})
                if trends:
                    st.markdown("**行业趋势**")
                    for k, v in trends.items():
                        st.markdown(f"- **{k}**：{v}")

        # --- 建模阶段 ---
        m = analysis.get("world_model")
        if m:
            with st.expander("② 建模 · 市场内部模型", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-cell"><div class="k">市场状态</div>'
                            f'<div class="v" style="font-size:0.9rem;">{m.get("market_state", "—")}</div></div>',
                            unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-cell"><div class="k">经济周期</div>'
                            f'<div class="v" style="font-size:0.9rem;">{m.get("economic_cycle", "—")}</div></div>',
                            unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-cell"><div class="k">市场情绪</div>'
                            f'<div class="v" style="font-size:0.9rem;">{m.get("market_sentiment", "—")}</div></div>',
                            unsafe_allow_html=True)
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**风险因素**")
                    for r in (m.get("risk_factors") or [])[:4]:
                        st.markdown(f"- {r}")
                with rc2:
                    st.markdown("**机会领域**")
                    for o in (m.get("opportunity_areas") or [])[:4]:
                        st.markdown(f"- {o}")

        # --- 推理阶段 ---
        plans = analysis.get("reasoning_plans")
        if plans:
            with st.expander("③ 推理 · 候选方案竞技场", expanded=False):
                for i, plan in enumerate(plans):
                    st.markdown(
                        f'<div class="card"><div class="card-title"><span class="bar"></span>'
                        f'方案 {i + 1} · {plan.get("plan_id", "")}</div>'
                        + score_bar("综合置信度", plan.get("confidence_level", 0))
                        + score_bar("数据支撑", plan.get("data_support_score", 0), "权重40%")
                        + score_bar("逻辑自洽", plan.get("logic_coherence_score", 0), "权重30%")
                        + score_bar("风险可控", plan.get("risk_controllability_score", 0), "权重30%")
                        + f'<div style="font-size:0.85rem;margin-top:0.5rem;">'
                        f'<b>假设</b>　{plan.get("hypothesis", "")}<br>'
                        f'<b>方法</b>　{plan.get("analysis_approach", "")}<br>'
                        f'<b>预期</b>　{plan.get("expected_outcome", "")}</div>',
                        unsafe_allow_html=True)
                    dv = plan.get("data_validation")
                    if dv:
                        change = dv.get("confidence_change", 0)
                        tag_cls = "red" if change < 0 else ("green" if change > 0 else "")
                        st.markdown(
                            f'<span class="tag {tag_cls}">数据校验 '
                            f'{dv.get("original_confidence", 0):.0%} → '
                            f'{dv.get("adjusted_confidence", 0):.0%}（{change:+.2f}）</span>',
                            unsafe_allow_html=True)
                        for c in dv.get("conflicts", []):
                            st.markdown(f'<span class="tag red">矛盾 · {c}</span>', unsafe_allow_html=True)
                        for c in dv.get("consistencies", []):
                            st.markdown(f'<span class="tag green">吻合 · {c}</span>', unsafe_allow_html=True)
                    ce = plan.get("cross_eval_results")
                    if ce:
                        st.caption(f"多模型交叉评估 · {ce.get('fusion_note', '')}")
                    st.markdown("</div>", unsafe_allow_html=True)

        # --- 决策阶段 ---
        dec = analysis.get("selected_plan")
        if dec:
            with st.expander("④ 决策 · 最优方案", expanded=True):
                st.markdown(
                    f'<div class="card" style="border-color:rgba(79,140,255,0.35);">'
                    f'<div class="card-title"><span class="bar"></span>选中 {dec.get("selected_plan_id", "")}'
                    f'</div>'
                    f'<div style="font-size:0.9rem;line-height:1.8;">'
                    f'<b>投资论点</b>　{dec.get("investment_thesis", "")}<br>'
                    f'<b>风险评估</b>　{dec.get("risk_assessment", "")}<br>'
                    f'<b>时间框架</b>　{dec.get("timeframe", "")}</div>'
                    f'<div style="margin-top:0.6rem;padding:0.7rem 1rem;border-radius:10px;'
                    f'background:rgba(79,140,255,0.1);border:1px solid rgba(79,140,255,0.3);">'
                    f'<b>建议</b>　{dec.get("recommendation", "")}</div></div>',
                    unsafe_allow_html=True)
                for e in (dec.get("supporting_evidence") or [])[:4]:
                    st.markdown(f"- {e}")

        # --- 反思阶段 ---
        ref = analysis.get("reflection")
        if ref:
            with st.expander("⑤ 反思 · Reflector 独立风控审核", expanded=False):
                passed = ref.get("passed", True)
                badge = ('<span class="tag green">审核通过</span>' if passed
                         else '<span class="tag red">审核未通过</span>')
                reviewer = ref.get("reviewer_model")
                reviewer_tag = (f'<span class="tag purple">审核模型 · {reviewer}</span>'
                                if reviewer else "")
                st.markdown(
                    f'{badge}<span class="tag">评分 {ref.get("score", 0):.2f}</span>{reviewer_tag}',
                    unsafe_allow_html=True)
                if ref.get("issues"):
                    st.markdown("**发现问题**")
                    for issue in ref["issues"]:
                        st.markdown(f"- {issue}")
                if ref.get("revision_hints"):
                    st.markdown("**修正意见（已回灌推理阶段）**")
                    for h in ref["revision_hints"]:
                        st.markdown(f"- {h}")
                if ref.get("lessons"):
                    st.markdown("**沉淀经验（已写入长期记忆）**")
                    for lesson in ref["lessons"]:
                        st.markdown(f"- {lesson}")

        # --- 执行概要 ---
        ts = analysis.get("trace_summary")
        if ts:
            cells = (
                f'<div class="metric-cell"><div class="k">执行步骤</div>'
                f'<div class="v">{ts.get("total_steps", 0)}</div></div>'
                f'<div class="metric-cell"><div class="k">总耗时</div>'
                f'<div class="v">{ts.get("total_latency_ms", 0) / 1000:.1f}s</div></div>'
                f'<div class="metric-cell"><div class="k">LLM Tokens</div>'
                f'<div class="v">{ts.get("total_llm_tokens", 0):,}</div></div>'
                f'<div class="metric-cell"><div class="k">工具调用</div>'
                f'<div class="v">{ts.get("total_tool_calls", 0)}</div></div>'
            )
            if ts.get("tool_success_rate") is not None:
                cells += (f'<div class="metric-cell"><div class="k">工具成功率</div>'
                          f'<div class="v">{ts["tool_success_rate"]:.0%}</div></div>')
            st.markdown(f'<div class="metric-grid" style="margin:0.8rem 0;">{cells}</div>',
                        unsafe_allow_html=True)

        # --- 最终报告 ---
        if analysis.get("final_report"):
            st.markdown('<div class="card-title" style="margin-top:1rem;"><span class="bar"></span>'
                        '最终投资报告</div>', unsafe_allow_html=True)
            st.caption(f"{analysis.get('research_topic', '')} · {analysis.get('industry_focus', '')} · "
                       f"{analysis.get('time_horizon', '')}")
            st.markdown(f'<div class="report-box">{analysis["final_report"]}</div>',
                        unsafe_allow_html=True)

            dl_col, regen_col, _ = st.columns([1, 1, 4])
            with dl_col:
                if st.session_state.pdf_data:
                    st.download_button(
                        "下载 PDF 报告", data=st.session_state.pdf_data,
                        file_name=f"投研报告_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf", use_container_width=True, type="primary")
                else:
                    if st.button("生成 PDF 报告", use_container_width=True, type="primary"):
                        with st.spinner("生成中..."):
                            try:
                                resp = requests.post(
                                    f"{API_BASE_URL}/api/v1/research/"
                                    f"{st.session_state.current_task_id}/export-pdf", timeout=60)
                                resp.raise_for_status()
                                st.session_state.pdf_data = resp.content
                                st.rerun()
                            except Exception as e:
                                st.error(f"PDF 生成失败：{e}")
            with regen_col:
                if st.button("清空结果", use_container_width=True):
                    st.session_state.current_analysis = None
                    st.session_state.current_task_id = None
                    st.session_state.pdf_data = None
                    st.session_state.live_events = []
                    st.rerun()
    elif not start_analysis:
        st.markdown(
            '<div class="card" style="text-align:center;padding:3rem 2rem;">'
            '<div style="font-size:1.1rem;font-weight:700;">准备就绪</div>'
            '<div style="color:var(--text-dim);font-size:0.86rem;margin-top:0.5rem;">'
            '在左侧配置研究主题与模型，点击「启动智能分析」<br>'
            'Agent 将自主规划数据采集、生成候选方案、交叉评估、独立风控审核并产出报告</div></div>',
            unsafe_allow_html=True)

# -------------------- Agent 观测台 --------------------
with tab_observe:
    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown('<div class="card-title"><span class="bar"></span>工具注册中心（动态发现）</div>',
                    unsafe_allow_html=True)
        tools_res = api_get("/api/v1/agent/tools")
        if tools_res.get("success"):
            tools = tools_res.get("data", [])
            st.caption(f"共 {len(tools)} 个工具 · 新增工具只需注册即自动可见")
            rows = "".join(
                f'<tr><td><code>{t["name"]}</code></td>'
                f'<td style="color:var(--text-dim);">{t.get("description", "")[:38]}</td>'
                f'<td><span class="tag {"green" if t.get("readonly") else "red"}">'
                f'{"只读" if t.get("readonly") else "可写"}</span></td>'
                f'<td>{t.get("timeout", 0)}s</td></tr>'
                for t in tools)
            st.markdown(
                f'<div class="card" style="max-height:420px;overflow-y:auto;">'
                f'<table class="obs"><tr><th>工具</th><th>说明</th><th>权限</th><th>超时</th></tr>'
                f'{rows}</table></div>', unsafe_allow_html=True)
        else:
            st.info("工具清单不可用")

        st.markdown('<div class="card-title"><span class="bar"></span>长期记忆</div>',
                    unsafe_allow_html=True)
        mem_res = api_get("/api/v1/agent/memory/stats")
        if mem_res.get("success"):
            md = mem_res.get("data", {})
            cells = (
                f'<div class="metric-cell"><div class="k">记忆条数</div>'
                f'<div class="v">{md.get("count", 0)}</div></div>'
                f'<div class="metric-cell"><div class="k">容量上限</div>'
                f'<div class="v">{md.get("max_items", 0)}</div></div>'
                f'<div class="metric-cell"><div class="k">Embedding</div>'
                f'<div class="v" style="font-size:0.85rem;">{md.get("embedding_mode", "—")}</div></div>'
            )
            st.markdown(f'<div class="metric-grid">{cells}</div>', unsafe_allow_html=True)
        else:
            st.info("记忆统计不可用")

    with oc2:
        st.markdown('<div class="card-title"><span class="bar"></span>MCP 协议状态</div>',
                    unsafe_allow_html=True)
        mcp_res = api_get("/api/v1/agent/mcp/status", timeout=45)
        if mcp_res.get("success"):
            md = mcp_res.get("data", {})
            if md.get("available"):
                st.markdown('<span class="tag green">MCP Server 在线</span>', unsafe_allow_html=True)
                discovered = md.get("discovered_tools", [])
                st.caption(f"tools/list 动态发现 {len(discovered)} 个工具")
                st.markdown("".join(f'<span class="tag">{t}</span>' for t in discovered),
                            unsafe_allow_html=True)
            else:
                st.markdown('<span class="tag amber">MCP 不可用 · 已降级进程内执行</span>',
                            unsafe_allow_html=True)
        else:
            st.info(mcp_res.get("message", "MCP 探测失败"))

        st.markdown('<div class="card-title" style="margin-top:1rem;"><span class="bar"></span>'
                    '工具级指标（能力边界）</div>', unsafe_allow_html=True)
        met_res = api_get("/api/v1/agent/tools/metrics")
        if met_res.get("success") and met_res.get("data"):
            rows = ""
            for name, s in met_res["data"].items():
                rate = s.get("success_rate", 0)
                rate_cls = "green" if rate >= 0.9 else ("amber" if rate >= 0.7 else "red")
                errs = ", ".join(f"{k}×{v}" for k, v in (s.get("error_distribution") or {}).items())
                rows += (f'<tr><td><code>{name}</code></td><td>{s.get("calls", 0)}</td>'
                         f'<td><span class="tag {rate_cls}">{rate:.0%}</span></td>'
                         f'<td>{s.get("avg_latency_ms", 0)}ms</td>'
                         f'<td>{s.get("p95_latency_ms", 0)}ms</td>'
                         f'<td style="color:var(--text-dim);">{errs or "—"}</td></tr>')
            st.markdown(
                f'<div class="card" style="max-height:420px;overflow-y:auto;">'
                f'<table class="obs"><tr><th>工具</th><th>调用</th><th>成功率</th>'
                f'<th>均值时延</th><th>P95</th><th>异常分布</th></tr>{rows}</table></div>',
                unsafe_allow_html=True)
        else:
            st.info("暂无工具调用指标（执行一次分析后产生）")

# -------------------- 执行轨迹 --------------------
with tab_trace:
    tc1, tc2 = st.columns([3, 1])
    with tc1:
        trace_task_id = st.text_input(
            "任务 ID", value=st.session_state.current_task_id or "",
            placeholder="输入 task_id 回放执行轨迹", label_visibility="collapsed")
    with tc2:
        load_trace = st.button("加载轨迹", use_container_width=True, type="primary")

    if load_trace and trace_task_id.strip():
        trace_res = api_get(f"/api/v1/research/{trace_task_id.strip()}/trace")
        if trace_res.get("success"):
            st.session_state["_trace_data"] = trace_res.get("data")
        else:
            st.session_state["_trace_data"] = None
            st.error(trace_res.get("error", "Trace 不存在"))

    trace = st.session_state.get("_trace_data")
    if trace:
        summary = trace.get("summary", {})
        cells = (
            f'<div class="metric-cell"><div class="k">步骤数</div>'
            f'<div class="v">{summary.get("total_steps", 0)}</div></div>'
            f'<div class="metric-cell"><div class="k">总耗时</div>'
            f'<div class="v">{summary.get("total_latency_ms", 0) / 1000:.1f}s</div></div>'
            f'<div class="metric-cell"><div class="k">LLM Tokens</div>'
            f'<div class="v">{summary.get("total_llm_tokens", 0):,}</div></div>'
            f'<div class="metric-cell"><div class="k">工具调用</div>'
            f'<div class="v">{summary.get("total_tool_calls", 0)}</div></div>'
            f'<div class="metric-cell"><div class="k">工具成功率</div>'
            f'<div class="v">{summary.get("tool_success_rate") or 0:.0%}</div></div>'
        )
        st.markdown(f'<div class="metric-grid" style="margin:0.8rem 0;">{cells}</div>',
                    unsafe_allow_html=True)

        for step in trace.get("steps", []):
            step_name = step.get("step", "")
            ok = step.get("status") == "ok"
            with st.expander(
                    f"{'●' if ok else '✕'} {PHASE_NAME.get(step_name, step_name)} · "
                    f"{step.get('latency_ms', 0) / 1000:.1f}s · "
                    f"LLM {len(step.get('llm_calls', []))} 次 · 工具 {len(step.get('tool_calls', []))} 次",
                    expanded=False):
                for call in step.get("llm_calls", []):
                    st.markdown(
                        f'<div class="trace-step{"" if ok else " error"}">'
                        f'<div style="font-size:0.85rem;"><b>{call.get("purpose", "")}</b> '
                        f'<span class="tag">{call.get("model", "")}</span>'
                        f'<span class="tag">prompt {call.get("prompt_tokens", 0):,}</span>'
                        f'<span class="tag">completion {call.get("completion_tokens", 0):,}</span>'
                        f'<span class="tag">{call.get("latency_ms", 0)}ms</span></div></div>',
                        unsafe_allow_html=True)
                    if call.get("input_preview"):
                        st.caption(f"输入：{call['input_preview'][:220]}")
                    if call.get("output_preview"):
                        st.caption(f"输出：{call['output_preview'][:220]}")
                for tc in step.get("tool_calls", []):
                    ok_tag = "green" if tc.get("ok") else "red"
                    st.markdown(
                        f'<span class="tag {ok_tag}">{tc.get("tool", "")}</span>'
                        f'<span class="tag">{tc.get("latency_ms", 0)}ms</span>'
                        + (f'<span class="tag amber">截断</span>' if tc.get("truncated") else "")
                        + (f'<span class="tag red">{tc.get("error_type")}</span>'
                           if tc.get("error_type") else ""),
                        unsafe_allow_html=True)
                    st.caption(f"参数：{json.dumps(tc.get('arguments', {}), ensure_ascii=False)}")
                if step.get("error"):
                    st.error(step["error"])
                if step.get("output_summary"):
                    st.caption(step["output_summary"])

# -------------------- 历史档案 --------------------
with tab_history:
    history = load_history()
    hc1, hc2 = st.columns([4, 1])
    hc1.caption(f"共 {len(history)} 条分析档案")
    with hc2:
        if history and st.button("清空全部", use_container_width=True):
            api_delete("/api/v1/history")
            st.rerun()

    if not history:
        st.info("暂无历史分析记录")
    for i, rec in enumerate(history):
        with st.container():
            st.markdown(
                f'<div class="card" style="padding:0.9rem 1.2rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div><b>{rec.get("topic", "")}</b>'
                f'<span style="color:var(--text-dim);font-size:0.78rem;margin-left:0.6rem;">'
                f'{rec.get("industry", "")} · {rec.get("horizon", "")} · {rec.get("model", "")}</span></div>'
                f'<div style="color:var(--text-dim);font-size:0.76rem;">{rec.get("time", "")}</div>'
                f'</div><div style="color:var(--text-dim);font-size:0.8rem;margin-top:0.4rem;">'
                f'{rec.get("report_preview", "")}</div></div>',
                unsafe_allow_html=True)
            dc, tc, _ = st.columns([1, 1, 6])
            with dc:
                if st.button("删除", key=f"hist_del_{i}", use_container_width=True):
                    api_delete(f"/api/v1/history/{i}")
                    st.rerun()
            with tc:
                if st.button("查看轨迹", key=f"hist_trace_{i}", use_container_width=True):
                    st.session_state["_trace_data"] = None
                    trace_res = api_get(f"/api/v1/research/{rec.get('id')}/trace")
                    if trace_res.get("success"):
                        st.session_state["_trace_data"] = trace_res.get("data")
                        st.info("轨迹已加载，切换到「执行轨迹」标签页查看")
                    else:
                        st.warning("该任务的 Trace 已清理")
