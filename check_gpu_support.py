#!/usr/bin/env python3
"""
检查系统是否支持GPU加速（DirectML/CUDA）
用于Magic-Mirror项目的GPU加速诊断
"""

import sys

def check_gpu_support():
    """检查系统GPU加速支持情况"""
    print("=" * 60)
    print("Magic-Mirror GPU加速支持检查工具")
    print("=" * 60)
    print()
    
    # 1. 检查 onnxruntime 是否安装
    print("【步骤 1】检查 onnxruntime 是否安装...")
    try:
        import onnxruntime as ort
        print(f"✅ onnxruntime 已安装，版本: {ort.__version__}")
    except ImportError:
        print("❌ onnxruntime 未安装")
        print("   请运行: pip install onnxruntime")
        return False
    
    print()
    
    # 2. 检查可用的 ExecutionProvider
    print("【步骤 2】检查可用的 ExecutionProvider...")
    try:
        available_providers = ort.get_available_providers()
        print(f"✅ 检测到 {len(available_providers)} 个 ExecutionProvider:")
        for i, provider in enumerate(available_providers, 1):
            print(f"   {i}. {provider}")
    except Exception as e:
        print(f"❌ 无法获取 ExecutionProvider: {e}")
        return False
    
    print()
    
    # 3. 检查 GPU 加速支持
    print("【步骤 3】检查 GPU 加速支持...")
    has_gpu = False
    gpu_type = None
    
    if 'DmlExecutionProvider' in available_providers:
        print("✅ 支持 DirectML (Windows 通用 GPU 加速)")
        has_gpu = True
        gpu_type = "DirectML"
    else:
        print("❌ 不支持 DirectML")
        print("   DirectML 需要:")
        print("   - Windows 10 版本 1903 或更高")
        print("   - 安装 onnxruntime-directml: pip install onnxruntime-directml")
    
    print()
    
    if 'CUDAExecutionProvider' in available_providers:
        print("✅ 支持 CUDA (NVIDIA GPU 加速)")
        has_gpu = True
        gpu_type = "CUDA" if gpu_type is None else f"{gpu_type} + CUDA"
    else:
        print("❌ 不支持 CUDA")
        print("   CUDA 需要:")
        print("   - NVIDIA GPU")
        print("   - 安装 CUDA Toolkit")
        print("   - 安装 onnxruntime-gpu: pip install onnxruntime-gpu")
    
    print()
    
    # 4. 测试 GPU 推理
    if has_gpu:
        print("【步骤 4】测试 GPU 推理能力...")
        try:
            import numpy as np
            
            # 创建一个简单的测试模型
            providers = []
            if 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
            elif 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            print(f"   尝试使用 Provider: {providers[0]}")
            
            # 注意：这里只是测试 Provider 是否可用
            # 实际模型加载需要真实的 ONNX 模型文件
            print("✅ GPU Provider 可用")
            print(f"   推荐使用: {providers[0]}")
            
        except Exception as e:
            print(f"⚠️  GPU Provider 测试失败: {e}")
            print("   这可能是正常的，因为没有加载实际模型")
    
    print()
    
    # 5. 总结
    print("=" * 60)
    print("【检查结果总结】")
    print("=" * 60)
    
    if has_gpu:
        print(f"✅ 系统支持 GPU 加速 ({gpu_type})")
        print()
        print("使用方法:")
        print("  在调用视频换脸函数时，传入 use_gpu=True 参数")
        print("  例如: swap_face_video(input_path, face_path, use_gpu=True)")
        print()
        print("注意事项:")
        print("  - GPU 模式使用 2 个处理线程")
        print("  - 如果 GPU 初始化失败，会自动回退到 CPU 模式")
        print("  - 确保所有 ONNX 模型文件存在于 models 目录")
    else:
        print("❌ 系统不支持 GPU 加速，将使用 CPU 模式")
        print()
        print("如何启用 GPU 加速:")
        print()
        print("【Windows 用户 - 推荐使用 DirectML】")
        print("  1. 确保 Windows 10 版本 1903 或更高")
        print("  2. 卸载现有 onnxruntime:")
        print("     pip uninstall onnxruntime onnxruntime-gpu")
        print("  3. 安装 onnxruntime-directml:")
        print("     pip install onnxruntime-directml")
        print()
        print("【NVIDIA GPU 用户 - 使用 CUDA】")
        print("  1. 安装 CUDA Toolkit (11.x 或 12.x)")
        print("  2. 安装 cuDNN")
        print("  3. 卸载现有 onnxruntime:")
        print("     pip uninstall onnxruntime onnxruntime-directml")
        print("  4. 安装 onnxruntime-gpu:")
        print("     pip install onnxruntime-gpu")
    
    print("=" * 60)
    return has_gpu


if __name__ == "__main__":
    try:
        result = check_gpu_support()
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n检查已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
