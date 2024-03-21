from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY","SERPER_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

def get_docs_text(docs):
    text=""
    for file in docs:
        loader = UnstructuredFileLoader(file)
        text += "Uploaded File: " + os.path.basename(file) + "\n\n" + loader.load()[0].page_content + "\n\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_db(text_chunks):
    db2 = Chroma.from_texts(text_chunks, embeddings, persist_directory="./chroma_db")
    db2.persist()

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=PROMPT)
    return chain

def user_input(user_question):
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = new_db.similarity_search(user_question,k=4)
    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    return response

def google_search(question):
    search = GoogleSerperAPIWrapper()
    ans = search.run(question)
    return ans

def get_headings(): 
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = new_db.as_retriever(score_threshold=0.7)

    prompt_template = """
    Given the text from the provided context, parse it to extract {question}. 
    Create a comprehensive list of important and conceptual titles from these headings. 
    Store the titles in list datatype of python and just return it in the way as provided below.
        ["title","title","title"]

    Exclude any non-heading text.\n\n

    Context:\n {context}?\n

    Answer:
    """

    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context","question"])

    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def get_faq():    
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = new_db.as_retriever(score_threshold=0.7)

    prompt_template = """
    Given the text from the provided context, parse it to extract important and conceptual {question} as much as possible. 
    Create and return a comprehensive list of question and answer in the way as provided below. 
            [["Question1: question?", "Answer: answer in brief"],
             ["Question2: question?", "Answer: answer in brief"],
             ["Question3: question?", "Answer: answer in brief"]]

    Context:\n {context}?\n

    Answer:
    """

    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context","question"])

    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def get_keyPW():
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = new_db.as_retriever(score_threshold=0.7)

    prompt_template = """
    Like a chapter in textbooks 
    provide a {question} in a dictionary and key points format 
    as in a learning management system 
    using a provided {context} and natural language processing
    Give the output as much as possible in the format given:
            ** KEYWORDS **
            word = definition  
            word = definition 
            word = definition 
            word = definition 
            ** KEY POINTS **
            - Summary point 1
            - Summary point 2
            - Summary point 3
    """

    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context","question"])

    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def get_quizz():    
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = new_db.as_retriever(score_threshold=0.7)

    prompt_template = """
    Given the text from the provided context, parse it to extract important and conceptual questions to form {question}. 
    Create and return only 10 multiple-choice questions in the format provided below:

    [[0,"question?",["option", "option", "option", "option"],"option"],
     [1,"question?",["option", "option", "option", "option"],"option"],
     [2,"question?",["option", "option", "option", "option"],"option"]]

    Context:
    {context}?

    Answer:
"""



    PROMPT = PromptTemplate(template = prompt_template, input_variables = ["context","question"])

    chain = RetrievalQA.from_chain_type(llm=model,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain
