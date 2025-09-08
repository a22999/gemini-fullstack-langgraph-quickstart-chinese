# =============================================================================
# LangGraph 聊天代理 - 混合模型工厂模块（共享版本）
# =============================================================================
# 本文件现在使用共享的模型工厂模块，保持向后兼容的接口：
# 1. 导入共享的混合模型功能
# 2. 提供向后兼容的函数别名
# 3. 保持原有的 API 接口不变
# 4. 统一模型管理方式
# =============================================================================

# 导入共享的模型工厂功能
from src.shared.model_factory import (
    create_mixed_chat_model,
    create_chat_model,
    create_gemini_model,
    create_siliconflow_model,
    validate_config,
    get_model_provider,
    get_model_info,
    list_all_model_info
)

# 导入共享的配置
from src.shared.configuration import ChatConfiguration



# =============================================================================
# 向后兼容的便捷函数
# =============================================================================

def create_query_generator_model(config=None, **kwargs):
    """【创建查询生成模型】创建用于查询生成的聊天模型。"""
    return create_mixed_chat_model("query_generator", config, **kwargs)


def create_reflection_model(config=None, **kwargs):
    """【创建反思评估模型】创建用于反思评估的聊天模型。"""
    return create_mixed_chat_model("reflection", config, **kwargs)


def create_answer_model(config=None, **kwargs):
    """【创建答案生成模型】创建用于答案生成的聊天模型。"""
    return create_mixed_chat_model("answer", config, **kwargs)


# create_chat_model 已从共享模块导入


# =============================================================================
# 向后兼容的别名函数
# =============================================================================

# 配置验证函数别名
validate_mixed_model_config = validate_config

# 提供商获取函数别名
get_mixed_model_provider = get_model_provider

# 模型信息函数已从共享模块导入：get_model_info, list_all_model_info