import os
import re
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings


# --- 設定 ---
HTML_DIR = "html_files"
INDEX_DIR = "faiss_index"

# .envを読み込む
load_dotenv()

# 環境変数から取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 1. HTMLファイル一覧取得 ---
def get_html_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".html")]

# --- 2. <article> と <h2> 抽出 ---
def extract_article_and_title(html_path):
    valid_tags = {'p', 'li', 'cite'}

    with open(html_path, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'lxml')

        # <article> 取得
        article = soup.find('article')

        # <time datetime=""> 取得
        time_tag = article.find('time')
        datetime = time_tag.get('datetime')

        # <h2> タイトル取得
        title_tag = article.find('h2')
        title = title_tag.get_text(strip=True) if title_tag else "（無題）"

        # すべての <section> を走査
        all_text = ''
        for section in article.find_all('section'):

            # 特定タグを抜粋
            for tag in section.descendants:
                if tag.name in valid_tags:
                    tag_text = tag.get_text(strip=True)

                    # 複数行を分割
                    text_lines = tag_text.splitlines()

                    # 各行を処理
                    for line_text in text_lines:

                        # 行頭の水平タブを削除
                        line_text = re.sub(r'^\t+', '', line_text, flags=re.MULTILINE)

                        # <li> タグに中黒を追加
                        if tag.name == 'li':
                            line_text = '・' + line_text

                        # <cite> タグをカッコでくくる
                        if tag.name == 'cite':
                            line_text = '（' + line_text + '）'

                        all_text += line_text + "\n"

                        # <p> の場合に改行を追加
                        if tag.name == 'p':
                            all_text += "\n"

            all_text += "\n"

        return title, datetime, all_text

# --- 3. Documentリスト化（metadataにtitleとfilenameを含む） ---
def build_documents_from_html(directory):
    documents = []
    for file_path in get_html_files(directory):
        title, datetime, all_text = extract_article_and_title(file_path)

        print(all_text)

        if all_text.strip():
            metadata = {
                "source": os.path.basename(file_path),
                "title": title,
                "datetime": datetime
            }
            doc = Document(page_content=all_text, metadata=metadata)
            documents.append(doc)
    return documents

# --- 4. 分割・ベクトル化・FAISS保存 ---
def build_vector_index(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    model_name = "intfloat/multilingual-e5-large"  # E5系の中でも日本語強いモデル
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

#    embeddings = OpenAIEmbeddings(
#        model="text-embedding-3-small",
#        openai_api_key=OPENAI_API_KEY
#    )

    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(INDEX_DIR)
    return db

# --- 5. 検索＆出典付き表示 ---
def search_with_query(db, query):
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        return_source_documents=True  # ← 出典を返すために必要
    )
    result = qa.invoke(query)

    answer = result['result']
    sources = result['source_documents']

    if sources:
        source_info = sources[0].metadata
        source_str = f"\n📚 出典: {source_info.get('title', '無題')}（{source_info.get('source', '不明')}）"
    else:
        source_str = "\n⚠️ 出典情報は見つかりませんでした。"

    return answer + source_str

# --- メイン処理 ---
if __name__ == "__main__":
    docs = build_documents_from_html(HTML_DIR)
    if not docs:
        print("❌ HTMLファイルからデータが抽出できませんでした。")
        exit()

    db = build_vector_index(docs)

    query = "山本慎一郎は情報セキュリティについてどのように考えていますか？　具体的、かつ、詳細に回答してください。"
    answer = search_with_query(db, query)
    print("🧠 回答:", answer)
