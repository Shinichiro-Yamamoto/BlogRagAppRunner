# ベースイメージ
FROM python:3.11-slim

# 必要なOSパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ作成
WORKDIR /app

# 必要ファイルをコピー
COPY . .

# Pythonライブラリインストール
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ポート開放
EXPOSE 8000

# アプリ起動
CMD ["uvicorn", "app_test:app", "--host", "0.0.0.0", "--port", "8000"]
