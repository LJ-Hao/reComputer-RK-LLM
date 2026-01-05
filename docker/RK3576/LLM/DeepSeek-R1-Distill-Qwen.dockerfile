# 显式声明平台，增加构建稳定性
FROM --platform=$TARGETPLATFORM python:3.10-slim AS base

RUN apt-get update && \
    apt-get install -y wget curl git sudo && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/models

COPY ./src/flask_server_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./src/librkllmrt.so /usr/lib/librkllmrt.so
COPY ./src/fix_freq_rk3576.sh /app/fix_freq_rk3576.sh
RUN chmod +x /app/fix_freq_rk3576.sh

# --- 模型构建阶段 ---
FROM base AS model-image
ARG MODEL_URL
ARG MODEL_FILE

# 使用 -O 确保下载到正确路径
RUN wget --progress=dot:giga "${MODEL_URL}" -O "/app/models/${MODEL_FILE}"

COPY ./src/flask_server.py /app/
ENV RKLLM_MODEL_PATH=/app/models/${MODEL_FILE}
EXPOSE 8080

# 建议使用 exec 格式的 CMD，能更好地处理系统信号
CMD ["sh", "-c", "python /app/flask_server.py --rkllm_model_path ${RKLLM_MODEL_PATH} --target_platform rk3576"]