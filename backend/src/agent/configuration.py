# =============================================================================
# LangGraph 智能研究代理 - 配置管理模块
# =============================================================================
# 本文件定义了整个研究代理的配置参数，实现了：
# 1. 模型配置：不同阶段使用的AI模型设置
# 2. 流程控制：研究循环和查询数量的限制参数
# 3. 环境集成：支持环境变量和运行时配置
# 4. 类型安全：使用 Pydantic 确保配置的有效性
# =============================================================================

# 系统模块
import os

# Pydantic 数据验证和设置管理
from pydantic import BaseModel, Field
from typing import Any, Optional

# LangChain 运行时配置
from langchain_core.runnables import RunnableConfig


# =============================================================================
# 配置类 - Configuration
# =============================================================================
# 智能研究代理的核心配置类，定义了所有可调参数
# 使用 Pydantic BaseModel 提供数据验证和序列化功能
# =============================================================================
class Configuration(BaseModel):
    """
    【智能研究代理配置】
    定义整个研究流程的核心配置参数。
    
    配置分为两大类：
    1. 模型配置：不同处理阶段使用的AI模型
    2. 流程配置：控制研究深度和广度的参数
    
    支持通过环境变量或运行时配置进行动态调整。
    """

    # =========================================================================
    # 模型配置部分
    # =========================================================================
    # 不同的处理阶段使用不同的模型，实现性能和成本的平衡
    
    # 查询生成模型：用于生成初始和后续搜索查询
    # 使用较快的模型以提高响应速度
    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "查询生成阶段使用的语言模型名称。负责根据用户问题生成搜索查询。"
        },
    )

    # 反思评估模型：用于评估研究充分性和生成后续查询
    # 使用平衡性能的模型进行复杂推理
    reflection_model: str = Field(
        default="gemini-2.5-flash",
        metadata={
            "description": "反思评估阶段使用的语言模型名称。负责判断研究完整性并生成改进建议。"
        },
    )

    # 答案生成模型：用于生成最终的研究报告
    # 使用最强的模型确保输出质量
    answer_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "答案生成阶段使用的语言模型名称。负责生成最终的综合研究报告。"
        },
    )

    # =========================================================================
    # 流程控制配置部分
    # =========================================================================
    # 控制研究的深度和广度，平衡质量和效率
    
    # 初始查询数量：第一轮生成的搜索查询数量
    # 影响研究的初始覆盖面
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "初始搜索查询的生成数量。数量越多覆盖面越广，但成本也越高。"},
    )

    # 最大研究循环次数：反思-搜索循环的最大执行次数
    # 防止无限循环，控制总体成本
    max_research_loops: int = Field(
        default=2,
        metadata={"description": "最大研究循环次数。每个循环包括反思评估和补充搜索。"},
    )

    # =========================================================================
    # 配置加载方法
    # =========================================================================
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        【配置实例化方法】
        从 LangChain 运行时配置创建 Configuration 实例。
        
        支持两种配置来源：
        1. 环境变量：使用大写的参数名作为环境变量名
        2. 运行时配置：通过 RunnableConfig 传递的配置字典
        
        优先级：环境变量 > 运行时配置 > 默认值
        
        参数:
            config: LangChain 运行时配置对象，可选
            
        返回:
            Configuration: 配置实例
        """
        # 步骤1: 提取可配置参数字典
        # 从 RunnableConfig 中安全提取 configurable 字段
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # 步骤2: 收集配置值
        # 遍历所有模型字段，从环境变量或配置中获取值
        # 环境变量名为字段名的大写形式（如 QUERY_GENERATOR_MODEL）
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # 步骤3: 过滤有效值
        # 移除 None 值，让 Pydantic 使用默认值
        values = {k: v for k, v in raw_values.items() if v is not None}

        # 步骤4: 创建并返回配置实例
        # Pydantic 会自动进行类型验证和转换
        return cls(**values)
