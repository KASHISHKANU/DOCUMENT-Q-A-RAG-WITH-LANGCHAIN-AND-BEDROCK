import os
import json
import boto3
import streamlit as st
import numpy as np

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")   # folder containing PDFs
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 512}
    )
    return llm

def get_llama_llm():
    llm = ChatBedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 512}
    )
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a detailed 
answer (at least 250 words). If you don't know the answer, say 
you don't know — don't make up an answer.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa.invoke({"query": query})
    return result["result"]

def main():
    st.set_page_config("Chat with PDF - AWS Bedrock", layout="centered")
    st.header("💬 Chat with PDF using AWS Bedrock Models")

    user_question = st.text_input("Ask a question based on your uploaded PDFs:")

    with st.sidebar:
        st.title("🧠 Vector Store Manager")
        if st.button("Update Vector Store"):
            with st.spinner("Processing your PDFs..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("✅ Vector store updated successfully!")

    if st.button("Claude 3 Sonnet Output"):
        if not user_question:
            st.warning("Please enter a question before querying.")
        else:
            with st.spinner("Querying Claude 3 Sonnet..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_claude_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.subheader("Claude 3 Response:")
                st.write(response)
                st.success("✅ Done")

    if st.button("Llama 3 Output"):
        if not user_question:
            st.warning("Please enter a question before querying.")
        else:
            with st.spinner("Querying Llama 3 70B..."):
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True
                )
                llm = get_llama_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.subheader("Llama 3 Response:")
                st.write(response)
                st.success("✅ Done")


if __name__ == "__main__":
    main()
