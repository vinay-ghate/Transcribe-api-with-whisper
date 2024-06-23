from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import BartTokenizer, BartForConditionalGeneration
import whisper
import torch
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

def transcribe_audio(file_path: str) -> str:
    model = whisper.load_model("large-v3")
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)
    return result.text

def summarize_text(text: str) -> str:
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_timestamps(transcript: str) -> list:
    words = transcript.split()
    timestamps = [(i*5, (i+1)*5, ' '.join(words[i*50:(i+1)*50])) for i in range(len(words) // 50)]
    return timestamps

app = FastAPI()

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    try:
        transcript = transcribe_audio(file_location)
        summary = summarize_text(transcript)
        timestamps = extract_timestamps(transcript)
    except Exception as e:
        os.remove(file_location)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    return JSONResponse(content={
        "transcript": transcript,
        "summary": summary,
        "timestamps": timestamps
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
