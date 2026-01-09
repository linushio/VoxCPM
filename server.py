import gc
import os
import time
import uuid
import soundfile as sf
import numpy as np
import threading
import logging
from typing import Optional
from flask import Flask, request, jsonify


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from funasr import AutoModel

from voxcpm import VoxCPM

# --- 全局变量 ---
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
tts_model: Optional[VoxCPM] = None
asr_model: Optional[AutoModel] = None
last_request_time = None
model_lock = threading.Lock()
release_timer = None
MODEL_RELEASE_DELAY = 30  # 30秒，单位：秒


# --- 输出目录 ---
OUTPUT_DIR = os.path.abspath("/data/dub-data/index-tts/generated_audio")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    logging.info(f"创建输出目录: {OUTPUT_DIR}")

def get_model():
    """
    获取TTS模型实例。
    如果模型未加载或模型类型改变，则进行初始化。这是一个线程安全的操作。
    
    Args:
        model_type: 模型类型，"voxcpm"
    """
    global tts_model
    
    # 使用双重检查锁定来提高性能，避免每次都获取锁
    if tts_model is None:
        with model_lock:
            # 再次检查，因为在等待锁的时候，其他线程可能已经加载了模型
            if tts_model is None:
                logging.info("开始加载 VoxCPM 模型...")
                try:
                    tts_model = VoxCPM.from_pretrained(
                        hf_model_id="openbmb/VoxCPM1.5",  # 或者使用本地路径
                        load_denoiser=True,
                        optimize=True
                    )
                    logging.info("VoxCPM 模型加载成功。")
                except Exception as e:
                    logging.error(f"加载 VoxCPM 模型时发生错误: {e}")
                    raise
    
    return tts_model


def get_asr_model() -> AutoModel:
    """
    懒加载 ASR 模型（用于从参考音频自动识别 prompt_text）。
    """
    global asr_model
    if asr_model is not None:
        return asr_model

    if asr_model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"开始加载 ASR 模型 SenseVoiceSmall，device={device} ...")
        try:
            asr_model = AutoModel(
                model="iic/SenseVoiceSmall",
                disable_update=True,
                log_level="WARNING",
                device=device,
            )
            logging.info("ASR 模型加载成功。")
        except Exception as e:
            logging.error(f"加载 ASR 模型失败: {e}")
            raise
    return asr_model


def recognize_prompt_text(prompt_wav_path: str) -> str:
    """
    使用 ASR 从参考音频中识别文本，用于自动生成 prompt_text。
    """
    model = get_asr_model()
    if prompt_wav_path is None:
        return ""
    res = model.generate(input=prompt_wav_path, language="auto", use_itn=True)
    text = res[0]["text"].split('|>')[-1]
    return text

def release_model():
    """
    释放模型和GPU内存。这是一个线程安全的操作。
    """
    global tts_model
    with model_lock:
        if tts_model is not None:
            logging.info(f"{MODEL_RELEASE_DELAY}秒内无请求，开始释放模型...")
            # 释放模型对象
            del tts_model
            tts_model = None
            gc.collect()
            # 如果使用torch和CUDA，清空CUDA缓存
            if 'torch' in globals() and hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU 缓存已清空。")
            logging.info("模型已成功释放。")

def schedule_model_release():
    """
    安排一个定时器，在指定延迟后释放模型。
    每次API调用都会重置此定时器。
    """
    global release_timer
    # 如果已有定时器，先取消
    if release_timer:
        release_timer.cancel()
    # 创建并启动新的定时器
    release_timer = threading.Timer(MODEL_RELEASE_DELAY, release_model)
    release_timer.start()
    logging.info(f"模型释放任务已安排在 {MODEL_RELEASE_DELAY} 秒后执行。")

def apply_fade_in_out(audio_data: np.ndarray,
                      sample_rate: int,
                      fade_in_ms: int = 110,
                      fade_out_ms: int = 110) -> np.ndarray:
    """
    对音频头部做线性淡入、末尾做线性淡出。
    audio_data: float(-1~1) 或 int16 PCM；支持 shape=(n,) 或 (n, ch)
    """
    # 转 float32（保持你的逻辑）
    if not np.issubdtype(audio_data.dtype, np.floating):
        audio_data = audio_data.astype(np.float32) / 32767.0
    else:
        audio_data = audio_data.astype(np.float32, copy=False)

    n = audio_data.shape[0]

    def _apply_fade(segment: np.ndarray, window: np.ndarray) -> np.ndarray:
        # 多通道广播
        if segment.ndim > 1:
            window = window.reshape(-1, 1)
        return segment * window

    # 头部淡入
    fade_in_samples = int(fade_in_ms / 1000 * sample_rate)
    if fade_in_samples > 0 and n >= fade_in_samples:
        fade_in_window = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        audio_data[:fade_in_samples] = _apply_fade(audio_data[:fade_in_samples], fade_in_window)

    # 尾部淡出
    fade_out_samples = int(fade_out_ms / 1000 * sample_rate)
    if fade_out_samples > 0 and n >= fade_out_samples:
        fade_out_window = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        audio_data[-fade_out_samples:] = _apply_fade(audio_data[-fade_out_samples:], fade_out_window)

    return audio_data

@app.route('/api/tts', methods=['POST'])
def text_to_speech_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的请求，请提供JSON格式的数据。"}), 400

    # 获取参数
    speaker_audio_file = data.get('speaker_audio_file')
    text = data.get('text')
    task_id = data.get('task_id') or str(uuid.uuid4())

    # 基础验证
    if not text:
        return jsonify({"error": "缺少必要参数 'text'。"}), 400

    # 如果提供了 speaker_audio_file，检查文件是否存在
    if speaker_audio_file and not os.path.exists(speaker_audio_file):
        return jsonify({"error": f"参考音频文件不存在: {speaker_audio_file}"}), 400

    logging.info(f"[tts] task_id={task_id} text='{text[:30]}...'")

    try:
        model = get_model()
        schedule_model_release()
        
        # 仍然用 model_lock 串行推理，保证非 thread-safe 模型安全
        with model_lock:
            # 输出文件名使用 task_id，便于排查
            output_filename = f"{task_id}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            start_time = time.time()
            
            # VoxCPM 参数
            prompt_text = data.get('prompt_text')  # 参考文本（可选，可以由后端 ASR 自动生成）
            cfg_value = data.get('cfg_value', 3.0)
            inference_timesteps = data.get('inference_timesteps', 30)
            normalize = data.get('normalize', False)
            denoise = data.get('denoise', False)

            # ---- 自动补全 prompt_text：只给了音频没给文本时，在服务端调用 ASR 生成 ----
            if speaker_audio_file and (not prompt_text or not str(prompt_text).strip()):
                try:
                    logging.info(f"[tts] task_id={task_id} 检测到仅提供参考音频，调用 ASR 自动识别 prompt_text ...")
                    prompt_text = recognize_prompt_text(speaker_audio_file)
                    if prompt_text:
                        logging.info(f"[tts] task_id={task_id} ASR 识别到的 prompt_text: {prompt_text[:50]}...")
                    else:
                        logging.warning(f"[tts] task_id={task_id} ASR 未能识别到有效文本，将退回自由创作模式")
                except Exception as e:
                    logging.error(f"[tts] task_id={task_id} ASR 识别失败，将退回自由创作模式: {e}")
                    prompt_text = None

            # ---- 保证 VoxCPM 的约束：prompt_wav_path 和 prompt_text 要么都提供，要么都为 None ----
            if not speaker_audio_file or not prompt_text or not str(prompt_text).strip():
                # 任一缺失或 prompt_text 为空字符串 -> 不走语音克隆，改为自由创作
                speaker_audio_file = None
                prompt_text = None

            # 生成音频
            wav = model.generate(
                text=text,
                prompt_wav_path=speaker_audio_file,
                prompt_text=prompt_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                denoise=denoise,
            )

            # 保存音频
            sample_rate = model.tts_model.sample_rate
            sf.write(output_path, wav, sample_rate)

            # 应用淡入淡出效果
            original_audio, sr = sf.read(output_path, dtype='float32')
            original_audio = apply_fade_in_out(original_audio, sr)
            sf.write(output_path, original_audio, sr)

            end_time = time.time()
            logging.info(f"[tts] task_id={task_id} done, cost={end_time - start_time:.2f}s")

        return jsonify({
            "message": "音频生成成功",
            "gen_audio_file": output_path,
            "task_id": task_id
        })

    except Exception as e:
        logging.error(f"[tts] task_id={task_id} error: {e}", exc_info=True)
        return jsonify({"error": f"处理请求时发生内部错误: {str(e)}", "task_id": task_id}), 500

if __name__ == '__main__':
    logging.info("启动Flask开发服务器...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False, processes=1)
