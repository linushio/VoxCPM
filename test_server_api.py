#!/usr/bin/env python3
"""
TTS Server API 测试脚本

使用方法:
    python test_server_api.py --model voxcpm --text "测试文本"
    python test_server_api.py --model indextts --text "测试文本" --ref-audio voice.wav
"""

import argparse
import requests
import json
import os
import time


def test_voxcpm(
    text: str,
    ref_audio: str = None,
    ref_text: str = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = False,
    denoise: bool = False,
    api_url: str = "http://localhost:5000/api/tts"
):
    """测试 VoxCPM 模型"""
    print(f"\n{'='*60}")
    print(f"🎙️ 测试 VoxCPM 模型")
    print(f"{'='*60}")
    
    data = {
        "model_type": "voxcpm",
        "text": text,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "normalize": normalize,
        "denoise": denoise,
    }
    
    if ref_audio:
        if not os.path.exists(ref_audio):
            print(f"❌ 错误：参考音频文件不存在: {ref_audio}")
            return
        data["speaker_audio_file"] = ref_audio
        data["prompt_text"] = ref_text if ref_text else ""
        print(f"📁 参考音频: {ref_audio}")
        if ref_text:
            print(f"📝 参考文本: {ref_text}")
    else:
        print(f"🎨 无参考音频（VoxCPM 自由创作模式）")
    
    print(f"📄 合成文本: {text}")
    print(f"⚙️ CFG Value: {cfg_value}")
    print(f"⚙️ Inference Steps: {inference_timesteps}")
    print(f"⚙️ Normalize: {normalize}")
    print(f"⚙️ Denoise: {denoise}")
    print(f"\n🚀 发送请求...")
    
    start_time = time.time()
    try:
        response = requests.post(api_url, json=data, timeout=300)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 成功！")
            print(f"⏱️ 耗时: {elapsed_time:.2f} 秒")
            print(f"📁 输出文件: {result['gen_audio_file']}")
            print(f"🆔 任务ID: {result['task_id']}")
        else:
            result = response.json()
            print(f"\n❌ 失败！")
            print(f"错误信息: {result.get('error', '未知错误')}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")


def test_indextts(
    text: str,
    ref_audio: str,
    num_beams: int = 30,
    temperature: float = 0.3,
    use_emo_text: bool = True,
    emo_alpha: float = 0.2,
    api_url: str = "http://localhost:5000/api/tts"
):
    """测试 IndexTTS 模型"""
    print(f"\n{'='*60}")
    print(f"🎙️ 测试 IndexTTS 模型")
    print(f"{'='*60}")
    
    if not os.path.exists(ref_audio):
        print(f"❌ 错误：参考音频文件不存在: {ref_audio}")
        return
    
    data = {
        "model_type": "indextts",
        "text": text,
        "speaker_audio_file": ref_audio,
        "num_beams": num_beams,
        "temperature": temperature,
        "use_emo_text": use_emo_text,
        "emo_text": text,
        "emo_alpha": emo_alpha,
    }
    
    print(f"📁 参考音频: {ref_audio}")
    print(f"📄 合成文本: {text}")
    print(f"⚙️ Num Beams: {num_beams}")
    print(f"⚙️ Temperature: {temperature}")
    print(f"⚙️ Use Emo Text: {use_emo_text}")
    print(f"⚙️ Emo Alpha: {emo_alpha}")
    print(f"\n🚀 发送请求...")
    
    start_time = time.time()
    try:
        response = requests.post(api_url, json=data, timeout=300)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ 成功！")
            print(f"⏱️ 耗时: {elapsed_time:.2f} 秒")
            print(f"📁 输出文件: {result['gen_audio_file']}")
            print(f"🆔 任务ID: {result['task_id']}")
        else:
            result = response.json()
            print(f"\n❌ 失败！")
            print(f"错误信息: {result.get('error', '未知错误')}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ 请求失败: {e}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TTS Server API 测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 测试 VoxCPM（无参考音频）:
   python test_server_api.py --model voxcpm --text "你好，欢迎使用 VoxCPM！"

2. 测试 VoxCPM（有参考音频）:
   python test_server_api.py --model voxcpm --text "你好，欢迎使用 VoxCPM！" \\
       --ref-audio examples/example.wav --ref-text "参考文本内容" --denoise

3. 测试 IndexTTS:
   python test_server_api.py --model indextts --text "你好，欢迎使用 IndexTTS！" \\
       --ref-audio examples/example.wav

4. 自定义参数:
   python test_server_api.py --model voxcpm --text "测试文本" \\
       --cfg-value 2.5 --inference-timesteps 15 --normalize
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["voxcpm", "indextts"],
        required=True,
        help="选择模型类型"
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        required=True,
        help="要合成的文本"
    )
    
    parser.add_argument(
        "--ref-audio",
        type=str,
        help="参考音频文件路径"
    )
    
    parser.add_argument(
        "--ref-text",
        type=str,
        help="参考音频对应的文本（仅用于 VoxCPM）"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:5000/api/tts",
        help="API 地址（默认: http://localhost:5000/api/tts）"
    )
    
    # VoxCPM 参数
    voxcpm_group = parser.add_argument_group("VoxCPM 参数")
    voxcpm_group.add_argument(
        "--cfg-value",
        type=float,
        default=2.0,
        help="CFG 引导强度（默认: 2.0）"
    )
    
    voxcpm_group.add_argument(
        "--inference-timesteps",
        type=int,
        default=10,
        help="推理步数（默认: 10）"
    )
    
    voxcpm_group.add_argument(
        "--normalize",
        action="store_true",
        help="启用文本正则化"
    )
    
    voxcpm_group.add_argument(
        "--denoise",
        action="store_true",
        help="启用参考音频降噪"
    )
    
    # IndexTTS 参数
    indextts_group = parser.add_argument_group("IndexTTS 参数")
    indextts_group.add_argument(
        "--num-beams",
        type=int,
        default=30,
        help="Beam search 数量（默认: 30）"
    )
    
    indextts_group.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="采样温度（默认: 0.3）"
    )
    
    indextts_group.add_argument(
        "--no-emo-text",
        action="store_true",
        help="禁用情感文本"
    )
    
    indextts_group.add_argument(
        "--emo-alpha",
        type=float,
        default=0.2,
        help="情感强度（默认: 0.2）"
    )
    
    args = parser.parse_args()
    
    # 执行测试
    if args.model == "voxcpm":
        test_voxcpm(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            normalize=args.normalize,
            denoise=args.denoise,
            api_url=args.api_url
        )
    
    elif args.model == "indextts":
        if not args.ref_audio:
            print("❌ 错误：IndexTTS 模型需要提供 --ref-audio 参数")
            return
        
        test_indextts(
            text=args.text,
            ref_audio=args.ref_audio,
            num_beams=args.num_beams,
            temperature=args.temperature,
            use_emo_text=not args.no_emo_text,
            emo_alpha=args.emo_alpha,
            api_url=args.api_url
        )


if __name__ == "__main__":
    main()

