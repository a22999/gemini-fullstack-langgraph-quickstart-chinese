# =============================================================================
# LangGraph 智能研究代理 - Agent 特有配置
# =============================================================================
# 本文件定义 agent 特有的流程控制参数，继承共享的模型配置功能
# =============================================================================

# 系统模块
import os
from typing import Any, Optional

# Pydantic 数据验证和设置管理
from pydantic import Field

# LangChain 运行时配置
from langchain_core.runnables import RunnableConfig

# 使用共享的配置基类
from src.shared.configuration import ChatConfiguration


# =============================================================================
# Agent 配置类
# =============================================================================
class Configuration(ChatConfiguration):
    """
    【智能研究代理配置】
    继承共享的混合模型配置，添加 agent 特有的流程控制参数。
    """

    # Agent 特有的流程控制参数
    number_of_initial_queries: int = Field(
        default_factory=lambda: int(os.getenv("NUMBER_OF_INITIAL_QUERIES", "5")),
        metadata={"description": "初始搜索查询的生成数量"},
    )

    max_research_loops: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCH_LOOPS", "3")),
        metadata={"description": "最大研究循环次数"},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 LangChain 运行时配置创建 Configuration 实例"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)
