# Load a medium article, split it into chunks, embed it, and store it in a vector database
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting GrowthMindsetBook")
    loader = TextLoader("output.txt")
    document = loader.load()
    print("Splitting into chunks")
    # Keep the chunck small enough so it will fit in the context window of the LLM
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(chunks)
    print(f"Splitted into {len(chunks)} chunks")
    print("Embedding chunks")
    embeddings = OpenAIEmbeddings()
    print("Ingesting into Pinecone")
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("Done!")
