# =============================================================================
# LangGraph 简单聊天代理 - 包初始化文件
# =============================================================================
# 本文件用于将chat目录标识为Python包，并导入核心组件
# 主要职责：
# 1. 导入LangGraph聊天图对象
# 2. 定义包的公共API
# 3. 控制可导入的对象
# =============================================================================

# 导入聊天图对象
from src.chat.graph import graph
from src.chat.dual_model_graph import dual_model_graph

# 导入核心组件
from src.chat.configuration import ChatConfiguration
from src.chat.state import ChatState, DualModelState

# 定义包的公共API
# 通过__all__列表控制 "from chat import *" 时导入的对象
__all__ = [
    "graph",
    "dual_model_graph", 
    "ChatConfiguration",
    "ChatState",
    "DualModelState"
]