"""
运行脚本: python test_multimodel.py -c [text|image|video|audio]
音频功能依赖安装（如需音频处理):
    pip install vllm[audio] --no-deps
    apt-get update
    apt-get install -y libsndfile1 ffmpeg
    pip install librosa audioread pooch
"""
import base64
import os
import requests
import tempfile
import subprocess
from openai import OpenAI

# import time
# from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# from openai import InternalServerError
# from vllm.utils import FlexibleArgumentParser
from argparse import ArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8901/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode('utf-8')

    return result


# Text-only inference
def run_text_only() -> None:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": "What's the capital of France?"
        }],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion.choices[0].message.content
    print("Chat completion output:", result)


# Single-image input inference
def run_single_image() -> None:
    # image_path = "./demo.jpg"
    image_path = "../../data/demo.jpg"
    print(f"{model=}")
    with open(image_path, "rb") as image_file:
        # 进行 Base64 编码
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please describe this image in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ],
        }],
        model=model,
        max_completion_tokens=1024,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded image:", result)

def preprocess_video_for_api(video_path, target_duration=10, target_fps=30):
    """
    预处理视频以适应API要求
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        # 截取前10秒并重新编码
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', str(target_duration),  # 截取时长
            '-r', str(target_fps),       # 设置帧率
            '-c:v', 'libx264',           # 视频编码
            '-preset', 'fast',           # 编码速度
            '-crf', '23',                # 质量参数
            '-pix_fmt', 'yuv420p',       # 像素格式
            '-c:a', 'aac',               # 音频编码
            '-b:a', '128k',              # 音频比特率
            '-movflags', '+faststart',   # 流式优化
            '-y',                        # 覆盖输出
            temp_path
        ]
        
        # 执行ffmpeg命令
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # 检查文件大小
        size = os.path.getsize(temp_path)
        # print(f"处理后的视频大小: {size / 1024 / 1024:.2f} MB")
        
        return temp_path
        
    except subprocess.CalledProcessError as e:
        print(f"视频处理失败: {e}")
        print(f"stderr: {e.stderr}")
        return video_path  # 如果失败，返回原文件

def run_video() -> None:
    file_path = "../../data/video.mp4"
    print(f"{model=}")
    
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return
    
    # 预处理视频
    # print("正在预处理视频...")
    processed_path = preprocess_video_for_api(file_path)
    
    try:
        with open(processed_path, 'rb') as file:
            file_content = file.read()
            video_base64 = base64.b64encode(file_content).decode('utf-8')
            # print(f"Base64编码大小: {len(video_base64)} 字符")
        
        # API调用
        chat_completion_from_base64 = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "what's in this video?"
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    }
                ]
            }],
            max_tokens=300,
        )
        
        result = chat_completion_from_base64.choices[0].message.content
        print("视频分析结果:", result)
        
    finally:
        # 清理临时文件
        if processed_path != file_path and os.path.exists(processed_path):
            os.unlink(processed_path)

def run_audio() -> None:

    file_path = "../../data/audio.mp3"
    
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return
    
    with open(file_path, 'rb') as file:
        audio_base64 = base64.b64encode(file.read()).decode('utf-8')
        print(f"Base64编码完成，长度: {len(audio_base64)}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请分析这段音频的内容"
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "mp3"
                        }
                    }
                ]
            }],
            max_tokens=300,
        )
        
        result = response.choices[0].message.content
        print(f"音频分析结果: {result}")
        
    except Exception as e:
        print(f"API调用失败: {e}")

example_function_map = {
    "text": run_text_only,
    "image": run_single_image,
    "video": run_video,
    "audio": run_audio,
}


def main(args) -> None:
    chat_type = args.chat_type
    example_function_map[chat_type]()

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Demo on using OpenAI client for online serving with '
        'multimodal language models served with vLLM.')
    parser.add_argument('--chat-type',
                        '-c',
                        type=str,
                        default="image",
                        choices=list(example_function_map.keys()),
                        help='Conversation type with multimodal data.')
    args = parser.parse_args()
    main(args)