
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()



with open("/Users/elizabethayankojo/Desktop/Rasa_Rag_TheraTalk/docs/CBT_Transcripts.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(raw_text)


embedding_model = OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY"))

vectorstore = FAISS.from_texts(chunks, embedding_model)


vectorstore.save_local("cbt_index")
print("Index built and saved.")
