import os
import tempfile
import uvicorn
import model
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# client = ChatGPTClient(OPENAI_API_KEY)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        custom_temp_dir = Path(__file__).parent.resolve() / "temp_audio"
        os.makedirs(custom_temp_dir.as_posix(), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=custom_temp_dir
        ) as tmp_audio_file:
            # Write the contents of the uploaded file to the temporary file
            tmp_audio_file.write(await file.read())
            tmp_audio_file.flush()

            # Get the path of the temporary file
            tmp_file_path = tmp_audio_file.name

            transcribed_text = model.asr(tmp_file_path)

            print('transc\n\n', transcribed_text)
            
            response = model.ask_model(transcribed_text)

            model.tts(response, tmp_audio_file.name)

            return FileResponse(tmp_audio_file.name, media_type="audio/wav")

    except Exception as e:
        return FileResponse(Path(__file__).parent / "error.mp3")


class ReqBody(BaseModel):
    text: str
    
@app.post("/text")
async def text_prompt(body: ReqBody):
    try:
        # Create a temporary file
        custom_temp_dir = Path(__file__).parent.resolve() / "temp_audio"
        os.makedirs(custom_temp_dir.as_posix(), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=custom_temp_dir
        ) as tmp_audio_file:

            response = model.ask_model(body.text)
            model.tts(response, tmp_audio_file.name)

            return FileResponse(tmp_audio_file.name, media_type="audio/wav")

    except Exception as e:
        return FileResponse(Path(__file__).parent / "error.mp3")
    
    # return {"response": model.ask_model(text)}

if __name__ == "__main__":
    uvicorn.run(app)
