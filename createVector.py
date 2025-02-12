from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import config


loader = DirectoryLoader(
    path=".",
    glob="sample.txt",
    loader_cls=TextLoader,
    loader_kwargs={'autodetect_encoding': True}
)
data = loader.load()
print(data)


text_splitter = CharacterTextSplitter(
    separator='=====\n',
    chunk_overlap=0,
    length_function=len
)
documents = text_splitter.create_documents([doc.page_content for doc in data])

with open("./output/text_chunks.txt", "w", encoding="utf-8") as file:
    for text in documents:
        file.write(text.page_content)
        file.write('\n--------------------------------------\n')


db = Chroma.from_documents(
    collection_name='collection_name_server',
    documents=documents,
    embedding=OpenAIEmbeddings(api_key=config.OPENAI_API_KEY),
    persist_directory='wdb'
)
if db:
    db.persist()
    db = None
else:
    print("Chroma DB has not been initialized.")
