# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# Streamlitの設定
st.set_page_config(page_title="Wフロントマニュアル", layout="wide")

# StreamlitのUI構築
st.title("📄 Wフロントアシスタント")
st.write("マニュアルに基づいた質問応答システムです。質問を入力してください。")

# OpenAIの設定
embeddings_model = OpenAIEmbeddings(api_key=st.secrets['openai']['OPENAI_API_KEY'], model="text-embedding-3-small")


llm = ChatOpenAI(api_key=st.secrets['openai']['OPENAI_API_KEY'],
                 model="gpt-4o-mini", temperature=0, max_tokens=200, top_p=0)


db = Chroma(collection_name='wdb',persist_directory="./wdb", embedding_function=embeddings_model)

prompt = PromptTemplate(
    input_variables=["document_snippet", "question"],
    template="""
あなたはドキュメントに基づいて質問に答えるアシスタントです。以下のドキュメントに基づいて質問に答えてください。質問者は質問形式で質問してこない可能性もあり、文章を汲み取って相手が知りたいことを返してください。
ドキュメントの内容はレジャーホテルのフロント業務に関するものです。できる限りドキュメントから情報を読み取り回答する意識をもってください。
もし記載にないことが問われた場合は、「わかりません。」を伝えてください。ただしあまりにも多くの質問に「わかりません。」と答えると、ユーザーに不親切だと思われるかもしれません。
文章は適切に返答し、適当な文章は返答しないようにしてください。

    ドキュメント：
    {document_snippet}

    質問：{question}

    答え：
    """
)



# チャットボット関数
def chatbot(question):
    question_embedding = embeddings_model.embed_query(question)
    document_snippet = db.similarity_search_by_vector(question_embedding, k=3)
    snippets = [doc.page_content for doc in document_snippet]
    snippets = "\n---\n".join(snippets)
    print(f'question: {question}')
    print(snippets)
    filled_prompt = prompt.format(document_snippet=snippets, question=question)
    response = llm.invoke(filled_prompt)
    return response.content, document_snippet

# ユーザー入力
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーの質問入力
question = st.chat_input("質問を入力してください...")
print(f'question: {question}')
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 回答生成
    response, document_snippet = chatbot(question)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
