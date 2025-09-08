# -*- coding: utf-8 -*-
"""
Agent包初始化文件 (Agent Package Initialization)

本文件是agent包的入口点，负责：
1. 导入核心的LangGraph图对象
2. 定义包的公共API接口
3. 控制从包中可以导入的对象

通过__all__列表，明确指定了graph作为包的主要导出对象，
这是整个研究代理的核心工作流图。
"""

from agent.graph import graph  # 导入LangGraph研究代理的核心图对象

# 定义包的公共API，只导出graph对象
# 这确保了包的接口清晰，用户只能访问预期的公共组件
__all__ = ["graph"]
