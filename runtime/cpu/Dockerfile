FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY runtime/cpu/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements.txt

COPY . /app

EXPOSE 8998

ENTRYPOINT ["python", "runtime/cpu/serve.py"]
CMD ["--host", "0.0.0.0", "--port", "8998"]
