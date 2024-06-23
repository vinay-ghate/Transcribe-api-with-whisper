# Audio Transcription and Summarization with FastAPI

This project provides a FastAPI-based server that handles audio files, transcribes them using the Whisper model, summarizes the content with the BART model, and extracts timestamps from the transcriptions.

## Requirements

Install the required libraries:

```bash
pip install fastapi uvicorn whisper transformers torch python-dotenv
```

## Usage

### Start the Server

Run the FastAPI server using:

```bash
uvicorn api_application:app --reload
```

### API Endpoint

#### POST `/process-audio/`

Upload an audio file to transcribe, summarize, and extract timestamps.

**Request:**

- Method: POST
- Content-Type: multipart/form-data
- Form Data:
  - `file`: The audio file to be processed (.wav, .mp3, etc.)

**Response:**

- `transcript`: The full transcription of the audio.
- `summary`: A summarized version of the transcription.
- `timestamps`: Extracted timestamps with corresponding transcript segments.

### Example

You can use `curl` to test the endpoint:

```bash
curl -X POST "http://localhost:8000/process-audio/" -F "file=@/path/to/your/audiofile.mp3"
```

## Project Structure

```plaintext
.
├── main.py          # Main FastAPI application
├── .env             # Environment variables (if needed)
├── README.md        # Project documentation
```

