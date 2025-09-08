#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双模型聊天系统演示脚本

这个脚本演示了如何使用双模型聊天系统：
1. 同时调用Gemini和硅基流动模型
2. 整合两个模型的回答
3. 返回综合性的答案

使用前请确保设置了正确的API密钥：
- GOOGLE_API_KEY: Gemini模型密钥
- SILICONFLOW_API_KEY: 硅基流动模型密钥
"""

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

from src.chat.dual_model_graph import dual_model_graph
from src.chat.state import DualModelState
from langchain_core.messages import HumanMessage

def check_api_keys():
    """检查API密钥是否设置"""
    # 检查多种可能的环境变量名称
    gemini_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    siliconflow_key = os.getenv('SILICONFLOW_API_KEY')
    
    print("🔑 API密钥检查:")
    print(f"Gemini API密钥: {'✅ 已设置' if gemini_key else '❌ 未设置'}")
    print(f"硅基流动API密钥: {'✅ 已设置' if siliconflow_key else '❌ 未设置'}")
    
    if gemini_key and siliconflow_key:
        # 设置标准的环境变量名称以确保兼容性
        os.environ['GOOGLE_API_KEY'] = gemini_key
        os.environ['SILICONFLOW_API_KEY'] = siliconflow_key
        return True
    
    print("\n⚠️  警告: 缺少API密钥，将无法进行实际的模型调用")
    print("请在.env文件中设置以下变量:")
    if not gemini_key:
        print("- GEMINI_API_KEY 或 GOOGLE_API_KEY")
    if not siliconflow_key:
        print("- SILICONFLOW_API_KEY")
    return False

def demo_dual_model_chat(question: str):
    """演示双模型聊天"""
    print(f"\n🤖 双模型聊天演示")
    print(f"问题: {question}")
    print("=" * 60)
    
    # 创建初始状态
    initial_state = DualModelState(
        messages=[HumanMessage(content=question)],
        processing_stage="initial"
    )
    
    try:
        # 执行双模型图
        print("🔄 正在调用双模型系统...")
        result = dual_model_graph.invoke(initial_state)
        
        print("\n📊 执行结果:")
        print(f"处理阶段: {result.get('processing_stage', '未知')}")
        
        # 显示各个模型的回答
        if result.get('gemini_response'):
            print("\n🟢 Gemini模型回答:")
            print("-" * 40)
            print(result['gemini_response'])
        
        if result.get('siliconflow_response'):
            print("\n🔵 硅基流动模型回答:")
            print("-" * 40)
            print(result['siliconflow_response'])
        
        # 显示整合后的答案
        if result.get('integrated_response'):
            print("\n🎯 整合后的综合答案:")
            print("=" * 40)
            print(result['integrated_response'])
        
        # 显示最终消息
        if result.get('messages') and len(result['messages']) > 1:
            final_message = result['messages'][-1]
            print("\n💬 最终回复:")
            print("=" * 40)
            print(final_message.content)
        
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 双模型聊天系统演示")
    print("=" * 60)
    
    # 检查API密钥
    has_keys = check_api_keys()
    
    if not has_keys:
        print("\n🔧 演示模式: 将使用模拟密钥进行测试（会产生API错误，但可以看到系统流程）")
        # 设置模拟密钥用于演示
        os.environ['GOOGLE_API_KEY'] = 'demo-key-for-testing'
        os.environ['SILICONFLOW_API_KEY'] = 'demo-key-for-testing'
    
    # 演示问题列表
    demo_questions = [
        "什么是人工智能？",
        "如何学习Python编程？",
        "解释一下机器学习的基本概念"
    ]
    
    print("\n📝 可选择的演示问题:")
    for i, q in enumerate(demo_questions, 1):
        print(f"{i}. {q}")
    
    try:
        choice = input("\n请选择问题编号 (1-3) 或输入自定义问题: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= 3:
            question = demo_questions[int(choice) - 1]
        else:
            question = choice if choice else demo_questions[0]
        
        # 执行演示
        demo_dual_model_chat(question)
        
    except KeyboardInterrupt:
        print("\n\n👋 演示已取消")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
    
    print("\n🎉 演示完成！")
    print("\n💡 提示:")
    print("- 设置真实API密钥后可以获得实际的模型回答")
    print("- 查看 DUAL_MODEL_README.md 了解更多使用方法")
    print("- 运行 dual_model_example.py 进行交互式体验")

if __name__ == "__main__":
    main()