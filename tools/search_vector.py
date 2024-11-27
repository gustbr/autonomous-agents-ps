import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
load_dotenv()


embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

if __name__ == "__main__":
    print("Searching in vector store")


def search_vector(query):
    """The book Growth Mindset is vectorized, search in the vector store and return the answer"""
    vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(input={"input": query})
    return result["answer"]


if __name__ == "__main__":
    print(search_vector("How many chapters are there in the book? List them."))
