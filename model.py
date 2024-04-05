import librosa
import torch
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, AutoProcessor
from easymms.models.tts import TTSModel

def asr(file_name):
    data = wavfile.read(file_name)

    model_id = "facebook/mms-1b-fl102"
    tokenizer = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    tokenizer.tokenizer.set_target_lang("amh")
    model.load_adapter("amh")

    input_audio, _ = librosa.load(file_name, sr=16000)
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = tokenizer.batch_decode(predicted_ids)[0]

    return text


def tts(text, file_name):

    # text = "በደም አፋሳሹ እና አስከፊው የትግራይ ጦርነት ምክንያት በተፈጠረ መቃቃርና አለመጣጣም የትግራይ የኦርቶዶክስ የሃይማኖት አባቶች ከኢትዮጵያ ኦሮቶዶክስ ቤተክርስትያን ተለይተው መንበረ ሰላማ በሚል የራሳቸው ቤተክህነት ማቋቋማቸው ይታወቃል።"

    tts = TTSModel('amh')
    from pathlib import Path
    tts.uroman_dir_path = Path("./uroman/bin")
    res = tts.synthesize(text)
    tts.save(res, file_name)

# Audio(res[0], rate=res[1])