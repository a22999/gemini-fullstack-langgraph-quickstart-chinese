#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双模型问答功能测试脚本（简化版）

本脚本用于测试新实现的双模型问答功能，验证：
1. 双模型图的正确执行
2. Gemini和硅基流动模型的并行调用
3. 回答整合功能
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ 已加载.env文件")
except ImportError:
    print("⚠️  python-dotenv未安装，尝试手动加载.env文件")
    # 手动加载.env文件
    env_file = project_root / '.env'
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ 手动加载.env文件完成")
    else:
        print("❌ .env文件不存在")

from langchain_core.messages import HumanMessage
from src.chat.graph import dual_model_chat_graph
from src.chat.state import DualModelState


async def test_simple_dual_model():
    """简化的双模型测试"""
    print("\n=== 简化双模型测试 ===")
    
    try:
        # 检查环境变量
        print("检查环境变量...")
        gemini_key = os.getenv('GEMINI_API_KEY')
        siliconflow_key = os.getenv('SILICONFLOW_API_KEY')
        
        print(f"GEMINI_API_KEY: {'已设置' if gemini_key else '未设置'}")
        print(f"SILICONFLOW_API_KEY: {'已设置' if siliconflow_key else '未设置'}")
        
        if not gemini_key or not siliconflow_key:
            print("❌ 缺少必要的API密钥，请检查环境变量配置")
            return False
        
        # 创建测试消息
        test_message = "你好，请简单介绍一下自己。"
        messages = [HumanMessage(content=test_message)]
        
        # 创建初始状态
        initial_state = DualModelState(
            messages=messages,
            processing_stage="initial",
            gemini_response=None,
            siliconflow_response=None,
            integrated_response=None
        )
        
        print(f"\n用户问题: {test_message}")
        print("开始执行双模型问答...")
        
        # 执行双模型问答图
        result = await dual_model_chat_graph.ainvoke(initial_state)
        
        # 输出结果
        print("\n=== 执行结果 ===")
        print(f"处理阶段: {result.get('processing_stage', 'unknown')}")
        
        gemini_resp = result.get('gemini_response', '无回答')
        siliconflow_resp = result.get('siliconflow_response', '无回答')
        integrated_resp = result.get('integrated_response', '整合失败')
        
        print(f"\n【Gemini模型回答】: {gemini_resp[:100]}{'...' if len(gemini_resp) > 100 else ''}")
        print(f"\n【硅基流动模型回答】: {siliconflow_resp[:100]}{'...' if len(siliconflow_resp) > 100 else ''}")
        print(f"\n【整合后的最终回答】: {integrated_resp[:100]}{'...' if len(integrated_resp) > 100 else ''}")
        
        # 检查是否成功
        if result.get('processing_stage') == 'completed':
            print("\n✅ 双模型问答测试成功！")
            return True
        else:
            print("\n❌ 双模型问答测试失败！")
            print(f"错误详情: {result}")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_individual_models():
    """测试单个模型调用"""
    print("\n=== 测试单个模型调用 ===")
    
    try:
        from src.shared.model_factory import create_gemini_model, create_siliconflow_model
        
        # 测试Gemini模型
        print("\n测试Gemini模型...")
        try:
            gemini_model = create_gemini_model(temperature=0.7)
            gemini_response = gemini_model.invoke([HumanMessage(content="你好")])
            print(f"Gemini回答: {gemini_response.content[:50]}...")
            print("✅ Gemini模型测试成功")
        except Exception as e:
            print(f"❌ Gemini模型测试失败: {str(e)}")
        
        # 测试硅基流动模型
        print("\n测试硅基流动模型...")
        try:
            siliconflow_model = create_siliconflow_model(temperature=0.7)
            siliconflow_response = siliconflow_model.invoke([HumanMessage(content="你好")])
            print(f"硅基流动回答: {siliconflow_response.content[:50]}...")
            print("✅ 硅基流动模型测试成功")
        except Exception as e:
            print(f"❌ 硅基流动模型测试失败: {str(e)}")
            
    except Exception as e:
        print(f"❌ 模型导入失败: {str(e)}")


async def main():
    """主测试函数"""
    print("🚀 开始双模型问答功能测试（简化版）")
    print("=" * 50)
    
    # 测试单个模型
    await test_individual_models()
    
    # 测试双模型图
    result = await test_simple_dual_model()
    
    # 总结测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    if result:
        print("🎉 双模型问答功能测试通过！")
    else:
        print("⚠️  测试未通过，请检查配置和实现")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())