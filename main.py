from langchainHelper import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    
class Question(BaseModel):
    question: str

@app.post("/answer")
async def provide_answer(question: Question):
    try:
        chain = get_qa_chain()
        response = chain(question.question)
        if "not" in response["result"] and "context" in response["result"]:
            googleAnswer = google_search(question.question)
            response["result"] += "\nThe following answer has been fetched from google:\n\n" + googleAnswer
            return {"answer": response["result"]}
        return {"answer": response["result"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
    
@app.post("/faq")
async def provide_faq():
    try:
        chain = get_faq()
        response = chain("Question and Answers")
        return {"QnA": response["result"]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})

