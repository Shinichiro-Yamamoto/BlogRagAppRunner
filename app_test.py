import asyncio
import os
import re
import openai

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, Tuple, List



# クエリリクエストクラス
class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description = 'ユーザーからの質問内容。検索対象とするテキスト。',
        example = 'AIセーフティとは何ですか？'
    )

class SourceInfo(BaseModel):
    title: str = Field(
        ...,
        description = '参照元記事のタイトル'
    )
    source: str = Field(
        ...,
        description = '参照元ファイル名'
    )
    datetime: str = Field(
        ...,
        description = '参照元ファイル日付'
    )

class SearchAnswerResponse(BaseModel):
    answer: str = Field(
        ...,
        description = '生成された最終回答文'
    )
    sources: List[SourceInfo] = Field(
        ...,
        description = '回答作成に使用した参照元のリスト'
    )



# エイリアス定数
QUERY_MODEL_NAME = 'gpt-4.1-nano' # 質問クエリのモデル名
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large' # ベクトル DB のモデル名
TEMPLATE_PARAM_NAME_CONTEXT = 'context_str'
TEMPLATE_PARAM_NAME_QUESTION = 'question'
PLACEHOLDER_PARAM_NAME_CONTEXT = '{' + TEMPLATE_PARAM_NAME_CONTEXT + '}'
PLACEHOLDER_PARAM_NAME_QUESTION = '{' + TEMPLATE_PARAM_NAME_QUESTION + '}'



# API待ち受けアプリケーション定義
app = FastAPI()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yamashin-riss.jp"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# OpenAI 用の API-KEY を取得
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MY_API_TOKEN = os.getenv('MY_API_TOKEN', 'default_token')



# ベクトルDB設定
# DB名
INDEX_DIR = 'faiss_index'

# DBモデル設定
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# DBロード
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)



# 本体RAG検索用LLM設定
llm = ChatOpenAI(
    model = QUERY_MODEL_NAME,
    openai_api_key = OPENAI_API_KEY,
    max_tokens = 4096,
    temperature = 0.2,
    top_p = 0.7
)



#================================================================
#
# 質問展開クエリ系
#
# ユーザから与えられた質問を元に、言い回しなどを変えた複数の質問に展開する
#
#================================================================

def regexp_response_pre_query(text: str) -> Optional[Tuple[str, str, str, str]]:
    """
    質問展開クエリの結果を、ひとつずつの質問に分割する.

    :param text: 質問展開クエリの結果文字列
    :type text: str

    :return: 分割された質問文字列のタプル、もしくは、None
    :rtype: Optional[Tuple[str, str, str, str]]
    """

    pattern = r"^(.+)\n\n(.+)\n\n(.+)\n\n(.+)$"
    match = re.match(pattern, text)
    
    if match:
        return match.groups()
    else:
        return None



def expand_query(req: QueryRequest) -> Optional[Tuple[str, str, str, str]]:
    """
    質問展開クエリを実行する.

    :param req: 本来の質問データ
    :type req: QueryRequest

    :return: 分割された事前クエリ文字列のタプル、もしくは、None
    :rtype: Optional[Tuple[str, str, str, str]]
    """

    # 本来の質問を元に、複数の質問に展開するプロンプトを生成
    custom_question = fr"""
以下の質問のニュアンスは変えずに、言い回しや使用する単語を変えてください。候補は4つ挙げて、それぞれを「\n\n」で区切って回答してください。
---
質問：
{req.question}
"""

    # OpenAI のクライアントを使って、質問展開クエリを実行
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model = QUERY_MODEL_NAME,
        messages = [
            {'role': 'system', 'content': 'あなたはRAG検索用に質問を多角的に展開するトークナイザーです。'},
            {'role': 'user', 'content': custom_question}
        ],
        temperature = 0.3,
        max_tokens = 500,
        top_p = 0.9
    )

    # 質問展開クエリの結果を質問単位に分割
    result = regexp_response_pre_query(response.choices[0].message.content)

    # 展開結果を返す
    if result:
        part1, part2, part3, part4 = result
        return (part1, part2, part3, part4)
    else:
        return (req.question, req.question, req.question, req.question)



#================================================================
#
# 質問クエリ系
#
# 展開した質問をひとつずつ実行する
#
#================================================================

def create_map_prompt() -> PromptTemplate:
    """
    Map ステージのプロンプトを生成する.

    :return: Map ステージのプロンプトテンプレート
    :rtype: PromptTemplate
    """

    # プロンプト定義
    map_prompt = PromptTemplate(
        input_variables = [TEMPLATE_PARAM_NAME_CONTEXT, TEMPLATE_PARAM_NAME_QUESTION],
        template = f"""
以下の文書に基づいて、質問に答えてください：
---
{PLACEHOLDER_PARAM_NAME_CONTEXT}
---
質問：{PLACEHOLDER_PARAM_NAME_QUESTION}
"""
    )

    return map_prompt



def create_reduce_prompt() -> PromptTemplate:
    """
    Reduce ステージのプロンプトを生成する.

    :return: Reduce ステージのプロンプトテンプレート
    :rtype: PromptTemplate
    """

    # プロンプト定義
    reduce_prompt = PromptTemplate(
        input_variables = [TEMPLATE_PARAM_NAME_CONTEXT, TEMPLATE_PARAM_NAME_QUESTION],
        template = f"""
以下は複数の文書に対する回答です。それらを統合し、重複を避けて一貫性のある最終回答を作成してください：
---
{PLACEHOLDER_PARAM_NAME_CONTEXT}
---
質問：{PLACEHOLDER_PARAM_NAME_QUESTION}
"""
    )

    return reduce_prompt



def create_map_chain(map_prompt: PromptTemplate) -> LLMChain:
    """
    Map ステージの LLM チェインを生成する.

    :return: Map ステージの LLM チェイン
    :rtype: LLMChain
    """

    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    return map_chain



def create_reduce_chain(reduce_prompt: PromptTemplate) -> LLMChain:
    """
    Reduce ステージの LLM チェインを生成する.

    :return: Reduce ステージの LLM チェイン
    :rtype: LLMChain
    """

    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    return reduce_chain



def get_question_retriever(faiss_db: FAISS):
    """
    質問の retriever を生成する.

    :param faiss_db: FAISS データベース
    :type faiss_db: FAISS

    :return: 質問の retriever
    :rtype: VectorStoreRetriever
    """

    retriever = faiss_db.as_retriever(
        search_type = 'mmr',
        search_kwargs = {'k': 10, 'lambda_mult': 0.5}
    )

    return retriever



def create_question_qa(map_chain: LLMChain, reduce_chain: LLMChain) -> MapReduceDocumentsChain:
    """
    質問の RetrievalQA を生成する.

    :param map_chain: Map ステージの LLM チェイン
    :type map_chain: LLMChain

    :param reduce_chain: Reduce ステージの LLM チェイン
    :type reduce_chain: LLMChain

    :return: 質問の RetrievalQA
    :rtype: MapReduceDocumentsChain
    """

    # Reduce ステージのチェインを生成（実態は Stuff チェイン）
    reduce_documents_chain = StuffDocumentsChain(
        llm_chain = reduce_chain,
        document_variable_name = TEMPLATE_PARAM_NAME_CONTEXT
    )

    # Map-Reduce 全体のチェインを生成
    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain = map_chain,
        reduce_documents_chain = reduce_documents_chain,
        document_variable_name = TEMPLATE_PARAM_NAME_CONTEXT,
        return_intermediate_steps = False
    )

    # 最終的な RetrievalQA を生成
    question_qa = RetrievalQA(
        retriever = get_question_retriever(db),
        combine_documents_chain = map_reduce_chain,
        return_source_documents = True
    )

    return question_qa



def get_question_qa() -> RetrievalQA:
    """
    質問の RetrievalQA を取得する.

    :return: 質問の RetrievalQA
    :rtype: RetrievalQA
    """

    map_prompt = create_map_prompt()

    reduce_prompt = create_reduce_prompt()

    map_chain = create_map_chain(map_prompt)

    reduce_chain = create_reduce_chain(reduce_prompt)

    question_qa = create_question_qa(map_chain, reduce_chain)

    return question_qa



def query_question_sync(qa: RetrievalQA, question: str) -> dict:
    """
    単一の質問を同期的に実行する.

    :param qa: 質問を処理する RetrievalQA
    :type qa: RetrievalQA

    :param question: 質問文字列
    :type question: str

    :return: 質問の結果
    :rtype: dict
    """

    return qa.invoke({"query": question})



async def query_question_async(qa: RetrievalQA, question: str) -> dict:
    """
    単一の質問を非同期的に実行する.

    :param qa: 質問を処理する RetrievalQA
    :type qa: RetrievalQA

    :param question: 質問文字列
    :type question: str

    :return: 質問の結果
    :rtype: dict
    """

    return await asyncio.to_thread(query_question_sync, qa, question)



async def query_questions(questions: Tuple[str, str, str, str]) -> Tuple[str, List[Document]]:
    """
    質問を実行する.

    :param questions: 質問文字列の配列
    :type questions: Tuple[str, str, str, str]

    :return: 質問の結果
    :rtype: Tuple[str, List[Document]]
    """

    answers = []
    sources = []

    # 質問の RetrievalQA を取得
    qa_reduce = get_question_qa()

    # すべての質問を並列に処理
    tasks = [
        query_question_async(qa_reduce, question)
        for question in questions
    ]
    results = await asyncio.gather(*tasks)

    # 回答を整形
    for result in results:
        answers.append(result['result'])
        sources.extend(result['source_documents'])  # 出典を蓄積

    # 回答を統合
    template = """
以下は、複数の質問に対する個別の回答です。各観点からの回答を統合し、全体的にまとめてください。

{answers}
"""

    compose_prompt = PromptTemplate.from_template(template)
    compose_chain = LLMChain(
        llm = llm,
        prompt = compose_prompt, 
        output_key = 'output_text'
    )

    answer = compose_chain.invoke({
        'answers': "\n\n".join(answers)
    })

    return answer, sources



#================================================================
#
# エントリポイント系
#
# 各種 API を構成する
#
#================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    HTTP 500 エラーをカスタマイズする.

    :param request: HTTPリクエスト
    :type request: Request

    :param exc: HTTP 500 例外
    :type exc: Exception
    """

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )



def verify_token(authorization: str = Header(...)):
    """
    トークン認証を行う.

    :param authorization: 認証リクエスト
    :type authorization: str
    """

    if authorization != f"Bearer {MY_API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")



@app.get(
    '/',
    summary = 'ヘルスチェック',
    description = 'AWS AppRunner からのヘルスチェック応答'
)
def health_check():
    return {"status": "ok"}



@app.post(
    '/documents/search',
    summary = 'ドキュメント検索',
    description = '入力されたクエリに関連する記事、資料を検索します。',
    response_model = SearchAnswerResponse
)
async def documents_search(req: QueryRequest, _: str = Depends(verify_token)):
    """
    質問を受け付け、回答する.

    :param req: 質問リクエスト
    :type req: QueryRequest

    :return: 質問の結果
    :rtype: Object
    """

    # 質問を展開
    questions = expand_query(req)

    # 本体クエリを実行
    answer, sources = await query_questions(questions)

    # 出典情報を整理
    source_info = []
    for doc in sources:
        meta = doc.metadata
        source_info.append({
            'title': meta.get('title', '無題'),
            'source': meta.get('source', '不明'),
            'datetime': meta.get('datetime', '不明')
        })

    # 応答を返す
    return {
        'answer': answer,
        'source': source_info
    }
