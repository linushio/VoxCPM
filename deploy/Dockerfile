# 使用支持CUDA的PyTorch基础镜像
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量解决常见问题
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml /app

# 安装Python依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

# 创建缓存目录
RUN mkdir -p /app/cache/huggingface /app/cache/torch /app/models

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 使用entrypoint脚本启动
ENTRYPOINT ["python3", "server.py"]
