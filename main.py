import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Streamlitの設定
st.set_page_config(page_title="Chatbot", layout="wide")

# OpenAIの設定
embeddings_model = OpenAIEmbeddings(api_key=st.secrets['openai']['OPENAI_API_KEY'], model="text-embedding-3-small")
llm = ChatOpenAI(api_key=st.secrets['openai']['OPENAI_API_KEY'], model="gpt-4o-mini")
db = Chroma(persist_directory="./wdb", embedding_function=embeddings_model)

# プロンプトテンプレート
template = """
あなたはドキュメントに基づいて質問に答えるアシスタントです。以下のドキュメントに基づいて質問に答えてください。
もし記載にないことが問われた場合は、わからないことを伝えてください。

ドキュメント：{document_snippet}

質問：{question}

答え：
"""
prompt = PromptTemplate(input_variables=["document_snippet", "question"], template=template)


# チャットボット関数
def chatbot(question):
    question_embedding = embeddings_model.embed_query(question)
    document_snippet = db.similarity_search_by_vector(question_embedding, k=2)
    filled_prompt = prompt.format(document_snippet=document_snippet, question=question)
    response = llm.invoke(filled_prompt)
    return response.content


# StreamlitのUI構築
st.title("📄 Wフロントアシスタント")
st.write("マニュアルに基づいた質問応答システムです。質問を入力してください。")

# ユーザー入力
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーの質問入力
question = st.chat_input("質問を入力してください...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 回答生成
    response = chatbot(question)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
