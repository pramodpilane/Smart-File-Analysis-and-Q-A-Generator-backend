import tempfile
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")


def get_docs_text(docs):
    text=""
    for file in docs:
        loader = UnstructuredFileLoader(file)
        text += loader.load()[0].page_content
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_db(text_chunks):
    # vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_qa_chain():    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(score_threshold=0.7)

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain
