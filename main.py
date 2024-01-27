import os
import tempfile
import uvicorn
import model
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

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

            return {"text": model.asr(tmp_file_path)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.0.172", port=8000)
