#!/usr/bin/env python3
# =============================================================================
# 双模型聊天测试脚本
# =============================================================================
# 本文件用于测试双模型聊天功能的基本流程
# 即使没有API密钥也能看到系统架构和错误处理
# =============================================================================

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_core.messages import HumanMessage
from src.chat.dual_model_graph import dual_model_graph
from src.chat.state import DualModelState


def test_dual_model_structure():
    """测试双模型图结构"""
    print("🔍 测试双模型图结构...")
    print(f"图类型: {type(dual_model_graph)}")
    print(f"图节点: {list(dual_model_graph.get_graph().nodes.keys())}")
    print(f"图边: {dual_model_graph.get_graph().edges}")
    print("✅ 图结构测试完成")


def test_state_structure():
    """测试状态结构"""
    print("\n🔍 测试状态结构...")
    
    # 创建测试状态
    test_state = {
        "messages": [HumanMessage(content="测试消息")],
        "gemini_response": None,
        "siliconflow_response": None,
        "integrated_response": None,
        "processing_stage": "initial"
    }
    
    print(f"状态字段: {list(test_state.keys())}")
    print(f"消息数量: {len(test_state['messages'])}")
    print(f"处理阶段: {test_state['processing_stage']}")
    print("✅ 状态结构测试完成")


def test_without_api_keys():
    """测试没有API密钥时的行为"""
    print("\n🔍 测试无API密钥情况...")
    
    # 临时清除API密钥
    original_gemini = os.environ.get("GEMINI_API_KEY")
    original_siliconflow = os.environ.get("SILICONFLOW_API_KEY")
    
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    if "SILICONFLOW_API_KEY" in os.environ:
        del os.environ["SILICONFLOW_API_KEY"]
    
    try:
        # 创建测试状态
        initial_state = {
            "messages": [HumanMessage(content="这是一个测试问题")],
            "processing_stage": "initial"
        }
        
        print("执行双模型图（预期会有配置错误）...")
        result = dual_model_graph.invoke(initial_state)
        
        print("\n📊 执行结果:")
        print(f"处理阶段: {result.get('processing_stage', '未知')}")
        
        if result.get("messages"):
            final_message = result["messages"][-1]
            print(f"返回消息: {final_message.content[:100]}...")
        
        print("✅ 错误处理测试完成")
        
    except Exception as e:
        print(f"❌ 测试异常: {str(e)}")
    
    finally:
        # 恢复原始API密钥
        if original_gemini:
            os.environ["GEMINI_API_KEY"] = original_gemini
        if original_siliconflow:
            os.environ["SILICONFLOW_API_KEY"] = original_siliconflow


def test_with_mock_api_keys():
    """使用模拟API密钥测试"""
    print("\n🔍 测试模拟API密钥情况...")
    
    # 设置模拟API密钥
    os.environ["GEMINI_API_KEY"] = "mock_gemini_key_for_testing"
    os.environ["SILICONFLOW_API_KEY"] = "mock_siliconflow_key_for_testing"
    
    try:
        # 创建测试状态
        initial_state = {
            "messages": [HumanMessage(content="什么是人工智能？")],
            "processing_stage": "initial"
        }
        
        print("执行双模型图（使用模拟密钥，预期会有API调用错误）...")
        result = dual_model_graph.invoke(initial_state)
        
        print("\n📊 执行结果:")
        print(f"处理阶段: {result.get('processing_stage', '未知')}")
        
        if result.get("messages"):
            final_message = result["messages"][-1]
            print(f"返回消息类型: {type(final_message)}")
            print(f"返回消息内容: {final_message.content[:200]}...")
        
        if result.get("gemini_response"):
            print(f"Gemini响应: {result['gemini_response'][:100]}...")
        
        if result.get("siliconflow_response"):
            print(f"硅基流动响应: {result['siliconflow_response'][:100]}...")
        
        print("✅ 模拟密钥测试完成")
        
    except Exception as e:
        print(f"❌ 测试异常: {str(e)}")
        import traceback
        traceback.print_exc()


def show_system_info():
    """显示系统信息"""
    print("\n📋 系统信息:")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    
    # 检查关键模块
    try:
        import langchain_core
        print(f"LangChain Core: ✅ {langchain_core.__version__ if hasattr(langchain_core, '__version__') else '已安装'}")
    except ImportError:
        print("LangChain Core: ❌ 未安装")
    
    try:
        import langgraph
        print(f"LangGraph: ✅ {langgraph.__version__ if hasattr(langgraph, '__version__') else '已安装'}")
    except ImportError:
        print("LangGraph: ❌ 未安装")
    
    try:
        import langchain_google_genai
        print("Google GenAI: ✅ 已安装")
    except ImportError:
        print("Google GenAI: ❌ 未安装")
    
    try:
        import langchain_openai
        print("OpenAI: ✅ 已安装")
    except ImportError:
        print("OpenAI: ❌ 未安装")


def main():
    """主测试函数"""
    print("🚀 双模型聊天系统测试")
    print("=" * 60)
    
    show_system_info()
    
    try:
        test_dual_model_structure()
        test_state_structure()
        test_without_api_keys()
        test_with_mock_api_keys()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试完成！")
        print("\n💡 使用说明:")
        print("1. 设置真实的API密钥后可以进行实际对话")
        print("2. 运行 dual_model_example.py 进行完整体验")
        print("3. 查看 DUAL_MODEL_README.md 了解详细使用方法")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()