"""Skills 体系 —— 方法论/流程性知识封装，按需加载注入 System Prompt

对齐 JD「Skills 体系」「构建业务 Agent、MCP 工具与 Skills 体系」：
- Skill 是可复用的 Markdown 文档，描述分析框架、适用场景、推荐工具组合
- 与 Tool（动作能力）互补：Tool 告诉模型"能做什么"，Skill 告诉模型"该怎么做"
- Agent 启动时扫描 skills/ 目录，任务开始时由模型/规则选择 Skill 注入
"""
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "skills")


@dataclass
class Skill:
    """技能定义"""
    name: str
    description: str
    content: str                      # Markdown 正文
    triggers: List[str] = field(default_factory=list)  # 触发关键词
    recommended_tools: List[str] = field(default_factory=list)


def _parse_skill_md(text: str, fallback_name: str) -> Skill:
    """解析 Skill Markdown：支持 YAML-like 头部元数据

    格式:
        ---
        name: value_investing
        description: 价值投资分析框架
        triggers: [价值, 基本面, 长期]
        recommended_tools: [get_stock_financial, get_stock_profit]
        ---
        # 正文...
    """
    meta: Dict[str, object] = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if m:
        header, body = m.group(1), m.group(2)
        for line in header.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if v.startswith("[") and v.endswith("]"):
                    meta[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
                else:
                    meta[k] = v.strip("'\"")

    return Skill(
        name=str(meta.get("name", fallback_name)),
        description=str(meta.get("description", "")),
        content=body.strip(),
        triggers=list(meta.get("triggers", []) or []),
        recommended_tools=list(meta.get("recommended_tools", []) or []),
    )


class SkillLoader:
    """Skill 加载与选择"""

    def __init__(self, skills_dir: str = SKILLS_DIR):
        self.skills_dir = skills_dir
        self._skills: Optional[Dict[str, Skill]] = None

    def load_all(self, force: bool = False) -> Dict[str, Skill]:
        if self._skills is not None and not force:
            return self._skills
        skills: Dict[str, Skill] = {}
        if not os.path.isdir(self.skills_dir):
            logger.warning(f"Skills 目录不存在: {self.skills_dir}")
            self._skills = skills
            return skills
        for fname in os.listdir(self.skills_dir):
            if fname.endswith(".md"):
                path = os.path.join(self.skills_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        skill = _parse_skill_md(f.read(), fname[:-3])
                    skills[skill.name] = skill
                except Exception as e:
                    logger.warning(f"加载 Skill 失败 {fname}: {e}")
        self._skills = skills
        logger.info(f"Skills 加载完成: {list(skills.keys())}")
        return skills

    def select(self, task_text: str) -> Optional[Skill]:
        """根据任务文本选择最匹配的 Skill（触发词命中数优先）"""
        skills = self.load_all()
        best: Optional[Skill] = None
        best_score = 0
        for skill in skills.values():
            score = sum(1 for t in skill.triggers if t and t in task_text)
            if score > best_score:
                best, best_score = skill, score
        if best is None and skills:
            # 无命中时返回第一个（默认方法论）
            best = next(iter(skills.values()))
        return best

    def render(self, skill: Skill) -> str:
        """渲染为可注入 System Prompt 的文本"""
        return (
            f"\n\n## 分析框架（Skill: {skill.name}）\n"
            f"{skill.description}\n\n"
            f"{skill.content}\n"
        )


skill_loader = SkillLoader()
