# =============================================================================
# LangGraph 共享模块 - 混合模型配置模块
# =============================================================================
# 本文件定义了统一的混合模型配置，实现了：
# 1. 多提供商支持：支持Gemini和硅基流动模型
# 2. 灵活配置：不同阶段可使用不同的模型提供商
# 3. 环境集成：支持环境变量和运行时配置
# 4. 类型安全：使用 Pydantic 确保配置的有效性
# 5. 向后兼容：支持 agent 和 chat 模块的不同使用方式
# =============================================================================

# 系统模块
import os

# Pydantic 数据验证和设置管理
from pydantic import BaseModel, Field
from typing import Any, Optional

# LangChain 运行时配置
from langchain_core.runnables import RunnableConfig


# =============================================================================
# 聊天配置类 - ChatConfiguration
# =============================================================================
# 统一的混合模型配置类，支持多提供商和灵活配置
# =============================================================================
class ChatConfiguration(BaseModel):
    """
    【统一混合模型配置】
    定义统一的模型配置参数，支持混合使用不同的模型提供商。
    
    功能特性：
    1. 多提供商支持：Gemini 和硅基流动
    2. 阶段特化：不同处理阶段可使用不同提供商
    3. 成本优化：根据需求选择合适的模型
    4. 性能平衡：在质量和速度之间找到最佳平衡
    5. 向后兼容：支持 agent 和 chat 模块的不同使用方式
    """

    # =========================================================================
    # 基础模型配置
    # =========================================================================
    
    # 默认模型提供商
    model_provider: str = Field(
        default_factory=lambda: os.getenv("MODEL_PROVIDER", "gemini"),
        metadata={
            "description": "默认模型提供商选择。支持 'gemini'（Google Gemini）或 'siliconflow'（硅基流动）。"
        },
    )
    
    # =========================================================================
    # 混合模型提供商配置
    # =========================================================================
    # 为不同的处理阶段配置独立的模型提供商
    
    # 查询生成阶段提供商
    query_generator_provider: str = Field(
        default_factory=lambda: os.getenv("QUERY_GENERATOR_PROVIDER", ""),
        metadata={
            "description": "查询生成模型的提供商。为空时使用默认 model_provider。"
        },
    )
    
    # 反思评估阶段提供商
    reflection_provider: str = Field(
        default_factory=lambda: os.getenv("REFLECTION_PROVIDER", ""),
        metadata={
            "description": "反思模型的提供商。为空时使用默认 model_provider。"
        },
    )
    
    # 答案生成阶段提供商
    answer_provider: str = Field(
        default_factory=lambda: os.getenv("ANSWER_PROVIDER", ""),
        metadata={
            "description": "答案生成模型的提供商。为空时使用默认 model_provider。"
        },
    )
    
    # 聊天对话阶段提供商
    chat_provider: str = Field(
        default_factory=lambda: os.getenv("CHAT_PROVIDER", ""),
        metadata={
            "description": "聊天对话模型的提供商。为空时使用默认 model_provider。"
        },
    )
    
    # =========================================================================
    # 模型名称配置
    # =========================================================================
    
    # 查询生成模型
    query_generator_model: str = Field(
        default_factory=lambda: os.getenv("QUERY_GENERATOR_MODEL", "gemini-2.0-flash"),
        metadata={
            "description": "查询生成阶段使用的语言模型名称。"
        },
    )
    
    # 反思评估模型
    reflection_model: str = Field(
        default_factory=lambda: os.getenv("REFLECTION_MODEL", "gemini-2.5-flash"),
        metadata={
            "description": "反思评估阶段使用的语言模型名称。"
        },
    )
    
    # 答案生成模型
    answer_model: str = Field(
        default_factory=lambda: os.getenv("ANSWER_MODEL", "gemini-2.5-pro"),
        metadata={
            "description": "答案生成阶段使用的语言模型名称。"
        },
    )
    
    # 聊天对话模型
    chat_model: str = Field(
        default_factory=lambda: os.getenv("CHAT_MODEL", "gemini-2.0-flash"),
        metadata={
            "description": "聊天对话使用的语言模型名称。"
        },
    )
    
    # =========================================================================
    # 硅基流动模型配置
    # =========================================================================
    
    # 硅基流动查询生成模型
    siliconflow_query_model: str = Field(
        default_factory=lambda: os.getenv("SILICONFLOW_QUERY_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        metadata={
            "description": "硅基流动查询生成模型名称。"
        },
    )
    
    # 硅基流动反思模型
    siliconflow_reflection_model: str = Field(
        default_factory=lambda: os.getenv("SILICONFLOW_REFLECTION_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        metadata={
            "description": "硅基流动反思评估模型名称。"
        },
    )
    
    # 硅基流动答案生成模型
    siliconflow_answer_model: str = Field(
        default_factory=lambda: os.getenv("SILICONFLOW_ANSWER_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        metadata={
            "description": "硅基流动答案生成模型名称。"
        },
    )
    
    # 硅基流动聊天模型
    siliconflow_chat_model: str = Field(
        default_factory=lambda: os.getenv("SILICONFLOW_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        metadata={
            "description": "硅基流动聊天对话模型名称。"
        },
    )
    
    # =========================================================================
    # 工厂方法
    # =========================================================================
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "ChatConfiguration":
        """
        【从运行时配置创建配置实例】
        根据 LangGraph 的运行时配置创建配置实例。
        
        Args:
            config: LangGraph 运行时配置对象
            
        Returns:
            ChatConfiguration: 配置实例
        """
        if config is None:
            return cls()
        
        # 从运行时配置中提取可配置参数
        configurable = config.get("configurable", {})
        
        # 创建配置实例，优先使用运行时配置，否则使用环境变量默认值
        return cls(
            # 基础配置
            model_provider=configurable.get("model_provider", os.getenv("MODEL_PROVIDER", "gemini")),
            
            # 混合提供商配置
            query_generator_provider=configurable.get("query_generator_provider", os.getenv("QUERY_GENERATOR_PROVIDER", "")),
            reflection_provider=configurable.get("reflection_provider", os.getenv("REFLECTION_PROVIDER", "")),
            answer_provider=configurable.get("answer_provider", os.getenv("ANSWER_PROVIDER", "")),
            chat_provider=configurable.get("chat_provider", os.getenv("CHAT_PROVIDER", "")),
            
            # 模型名称配置
            query_generator_model=configurable.get("query_generator_model", os.getenv("QUERY_GENERATOR_MODEL", "gemini-2.0-flash")),
            reflection_model=configurable.get("reflection_model", os.getenv("REFLECTION_MODEL", "gemini-2.5-flash")),
            answer_model=configurable.get("answer_model", os.getenv("ANSWER_MODEL", "gemini-2.5-pro")),
            chat_model=configurable.get("chat_model", os.getenv("CHAT_MODEL", "gemini-2.0-flash")),
            
            # 硅基流动模型配置
            siliconflow_query_model=configurable.get("siliconflow_query_model", os.getenv("SILICONFLOW_QUERY_MODEL", "Qwen/Qwen2.5-7B-Instruct")),
            siliconflow_reflection_model=configurable.get("siliconflow_reflection_model", os.getenv("SILICONFLOW_REFLECTION_MODEL", "Qwen/Qwen2.5-14B-Instruct")),
            siliconflow_answer_model=configurable.get("siliconflow_answer_model", os.getenv("SILICONFLOW_ANSWER_MODEL", "Qwen/Qwen2.5-72B-Instruct")),
            siliconflow_chat_model=configurable.get("siliconflow_chat_model", os.getenv("SILICONFLOW_CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")),
        )
    
    def get_model_for_stage(self, stage: str) -> tuple[str, str]:
        """
        【获取指定阶段的模型配置】
        根据阶段名称获取对应的模型提供商和模型名称。
        
        Args:
            stage: 阶段名称 ('query_generator', 'reflection', 'answer', 'chat')
            
        Returns:
            tuple: (提供商名称, 模型名称)
        """
        # 获取阶段特定的提供商，如果为空则使用默认提供商
        provider_map = {
            "query_generator": self.query_generator_provider or self.model_provider,
            "reflection": self.reflection_provider or self.model_provider,
            "answer": self.answer_provider or self.model_provider,
            "chat": self.chat_provider or self.model_provider,
        }
        
        provider = provider_map.get(stage, self.model_provider)
        
        # 根据提供商和阶段获取模型名称
        if provider.lower() == "siliconflow":
            model_map = {
                "query_generator": self.siliconflow_query_model,
                "reflection": self.siliconflow_reflection_model,
                "answer": self.siliconflow_answer_model,
                "chat": self.siliconflow_chat_model,
            }
        else:  # gemini
            model_map = {
                "query_generator": self.query_generator_model,
                "reflection": self.reflection_model,
                "answer": self.answer_model,
                "chat": self.chat_model,
            }
        
        model_name = model_map.get(stage, self.chat_model)
        
        return provider, model_name


# =============================================================================
# 向后兼容性支持
# =============================================================================
# 为 agent 模块提供兼容的 Configuration 类
# =============================================================================

class Configuration(ChatConfiguration):
    """
    【向后兼容的配置类】
    为 agent 模块提供向后兼容的配置接口。
    继承自 ChatConfiguration，保持所有功能的同时提供简化的接口。
    """
    pass