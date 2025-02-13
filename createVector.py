import re

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import config


def normalize_separators(text, separator="\n=====\n"):
    """
    テキスト中の区切り部分（「=====」を含む行）を正規化して、
    必ずseparatorの形にする。
    """
    # 例: 「=====」の前後にある改行や空白を除去し、正確に "\n=====\n" に置換する
    pattern = r'\n*\s*={5,}\s*\n*'
    normalized_text = re.sub(pattern, separator, text)
    return normalized_text


def clean_and_normalize_file(input_file, output_file, separator="\n=====\n"):
    # 入力ファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 複数の改行を1つの改行に統一し、各行の前後の空白を削除
    text = re.sub(r'\n+', '\n', text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_text = "\n".join(lines).strip()

    # separatorを正規化して挿入（必ずseparatorの形にする）
    normalized_text = normalize_separators(cleaned_text, separator)

    # 末尾にseparatorがない場合は追加（分割が確実に発生するように）
    if not normalized_text.endswith(separator):
        normalized_text += separator

    # 結果を出力ファイルに保存する
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(normalized_text)


# sample.txt はそのままにして、正規化したテキストを sample_clean.txt に保存する
clean_and_normalize_file('sample.txt', 'sample_clean.txt')

# sample_clean.txt を DirectoryLoader で読み込む
loader = DirectoryLoader(
    path=".",
    glob="sample_clean.txt",
    loader_cls=TextLoader,
    loader_kwargs={'autodetect_encoding': True}
)
data = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=100,
    separator='=====',
    chunk_overlap=0,
    length_function=len
)

documents = text_splitter.create_documents([doc.page_content for doc in data])

with open("./output/text_chunks.txt", "w", encoding="utf-8") as file:
    for text in documents:
        file.write(text.page_content)
        file.write('\n--------------------------------------\n')


db = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(api_key=config.OPENAI_API_KEY),
    persist_directory='wdb'
)

if db:
    db.persist()
    db = None
else:
    print("Chroma DB has not been initialized.")
