#!/usr/bin/env python3
# =============================================================================
# 双模型聊天示例
# =============================================================================
# 本文件演示如何使用双模型聊天图进行智能对话
# 功能：同时调用Gemini和硅基流动模型，然后整合答案
# =============================================================================

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 加载.env文件
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ 已加载环境变量文件: {env_path}")
else:
    print(f"⚠️  未找到环境变量文件: {env_path}")
    print(f"请在 {env_path} 中配置API密钥")

from langchain_core.messages import HumanMessage
from src.chat.dual_model_graph import dual_model_graph
from src.chat.state import DualModelState


def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查多种可能的环境变量名称
    gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    siliconflow_key = os.getenv('SILICONFLOW_API_KEY')
    
    missing_vars = []
    
    if gemini_key:
        print("✅ Gemini API密钥 已设置")
        # 设置标准的环境变量名称以确保兼容性
        os.environ['GOOGLE_API_KEY'] = gemini_key
    else:
        print("❌ Gemini API密钥 未设置")
        missing_vars.append(('GEMINI_API_KEY', 'Google Gemini API密钥'))
    
    if siliconflow_key:
        print("✅ SILICONFLOW_API_KEY 已设置")
    else:
        print("❌ SILICONFLOW_API_KEY 未设置")
        missing_vars.append(('SILICONFLOW_API_KEY', '硅基流动API密钥'))
    
    if missing_vars:
        print("\n❌ 环境配置不完整，请设置必要的API密钥")
        print("\n📋 需要设置的环境变量:")
        for var, desc in missing_vars:
            print(f"- {var}: {desc}")
        print("\n💡 可以在 .env 文件中设置这些变量")
        return False
    
    print("\n✅ 环境配置完整")
    return True


def run_dual_model_chat(question: str):
    """运行双模型聊天"""
    print(f"\n🤖 双模型聊天开始")
    print(f"📝 问题: {question}")
    print("="*60)
    
    # 创建初始状态
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "processing_stage": "initial"
    }
    
    try:
        print("⏳ 正在并行调用Gemini和硅基流动模型...")
        
        # 执行双模型图
        result = dual_model_graph.invoke(initial_state)
        
        print("\n📊 执行结果:")
        print(f"处理阶段: {result.get('processing_stage', '未知')}")
        
        # 显示最终回答
        if result.get("messages"):
            final_message = result["messages"][-1]
            print("\n🎯 最终整合回答:")
            print("-" * 40)
            print(final_message.content)
        
        # 显示详细信息
        if result.get("gemini_response"):
            print("\n🔵 Gemini原始回答:")
            print("-" * 30)
            print(result["gemini_response"])
        
        if result.get("siliconflow_response"):
            print("\n🟡 硅基流动原始回答:")
            print("-" * 30)
            print(result["siliconflow_response"])
        
        if result.get("integrated_response"):
            print("\n🔄 整合后回答:")
            print("-" * 30)
            print(result["integrated_response"])
            
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()


def interactive_chat():
    """交互式聊天模式"""
    print("\n🎮 进入交互式双模型聊天模式")
    print("输入 'quit' 或 'exit' 退出")
    print("="*60)
    
    while True:
        try:
            question = input("\n👤 您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
            
            if not question:
                print("请输入一个问题")
                continue
            
            run_dual_model_chat(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


def main():
    """主函数"""
    print("🚀 双模型聊天示例")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境配置不完整，请设置必要的API密钥")
        print("\n📋 需要设置的环境变量:")
        print("- GEMINI_API_KEY: Google Gemini API密钥")
        print("- SILICONFLOW_API_KEY: 硅基流动API密钥")
        print("\n💡 可以在 .env 文件中设置这些变量")
        return
    
    # 运行示例问题
    example_questions = [
        "什么是人工智能？",
        "Python和JavaScript的主要区别是什么？",
        "如何学习机器学习？"
    ]
    
    print("\n🎯 运行示例问题:")
    for i, question in enumerate(example_questions, 1):
        print(f"\n📌 示例 {i}/{len(example_questions)}")
        run_dual_model_chat(question)
        
        if i < len(example_questions):
            input("\n⏸️  按回车键继续下一个示例...")
    
    # 询问是否进入交互模式
    print("\n" + "="*60)
    choice = input("🤔 是否进入交互式聊天模式？(y/n): ").strip().lower()
    
    if choice in ['y', 'yes', '是', 'Y']:
        interactive_chat()
    else:
        print("👋 感谢使用双模型聊天示例！")


if __name__ == "__main__":
    main()