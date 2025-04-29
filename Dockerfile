# ─────────────────────────────────────────────
#  Base image
#  slim を使い、サイズを抑えつつビルドに必要なツールを追加
# ─────────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────────
#  OS-level build dependencies
#  tokenizers が Rust を要求するため rustc/cargo, build-essential を追加
# ─────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        rustc \
        cargo && \
    rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
#  Workdir
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
#  Copy source code & requirements
# ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ソースツリーと FAISS インデックスなどを全部コピー
COPY . .

# ─────────────────────────────────────────────
#  Hugging Face Spaces は 8080 番を探しに来る
# ─────────────────────────────────────────────
EXPOSE 8080

# ─────────────────────────────────────────────
#  Health-checkエンドポイントがあると緑ランプが早い
# ─────────────────────────────────────────────
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

# ─────────────────────────────────────────────
#  Start command  (Spaces は ENTRYPOINT/CMD をそのまま実行)
# ─────────────────────────────────────────────
CMD ["uvicorn", "app_test:app", "--host", "0.0.0.0", "--port", "8080"]
