# ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリ作成
WORKDIR /app

# 必要ファイルをすべてコピー
COPY . .

# ライブラリインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポート開放（FastAPIは8000番）
EXPOSE 8000

# 起動コマンド
CMD ["uvicorn", "app_test:app", "--host", "0.0.0.0", "--port", "8000"]
