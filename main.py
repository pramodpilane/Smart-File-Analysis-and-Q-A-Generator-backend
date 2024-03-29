from langchainHelper import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import random

app = FastAPI()

# Allowing CORS for all origins in development, you should restrict it in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to save uploaded files
UPLOAD_DIRECTORY = "bin"

# Ensure that the directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        file_locations=[]
        for file in files:
            file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
            with open(file_location, "wb") as file_object:
                file_object.write(await file.read())
            file_locations.append(file_location)
        print(file_locations)
        raw_text = get_docs_text(file_locations)  
        text_chunks = get_text_chunks(raw_text)
        create_vector_db(text_chunks)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to upload files: {str(e)}"})

@app.get("/check_bin_folder")
async def check_bin_folder():
    bin_folder_path = "C:/Users/user/OneDrive/Desktop/internshipProject/Smart-File-Analysis-and-Q-A-Generator/backend/bin" # Change this to the actual path
    if os.path.exists(bin_folder_path) and os.listdir(bin_folder_path):
        return {"bin_folder_has_files": True}
    else:
        return {"bin_folder_has_files": False}
  
class Question(BaseModel):
    question: str

@app.post("/answer")
async def provide_answer(question: Question):
    try:
        response = user_input(question.question)
        if "not" in response["output_text"] and "context" in response["output_text"]:
            googleAnswer = google_search(question.question)
            response["output_text"] += "\nThe following answer has been fetched from google:\n\n" + googleAnswer
            return {"answer": response["output_text"]}
        return {"answer": response["output_text"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    
@app.get("/quiz")
async def provide_faq():
    try:
        chain = get_quizz()
        response = chain("multiple choice questions")
        return {"quiz": response["result"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    
@app.get("/faq")
async def provide_faq():
    try:
        chain = get_faq()
        response = chain("Question and Answers")
        return {"QnA": response["result"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    
@app.get("/suggest")
async def provide_suggestion():
    try:
        chain = get_headings()
        topics = eval(chain("headings")["result"].strip())
        return {"suggestion": random.choice(topics)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})

@app.get("/keypw")
async def provide_faq():
    try:
        chain = get_keyPW()
        response = chain("summary")
        return {"keyPW": response["result"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
