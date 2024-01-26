import tempfile
import os
import model
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio_file:
            # Write the contents of the uploaded file to the temporary file
            tmp_audio_file.write(await file.read())
            tmp_audio_file.flush()

            # Get the path of the temporary file
            tmp_file_path = tmp_audio_file.name

            return {"text": model.asr(tmp_file_path)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
