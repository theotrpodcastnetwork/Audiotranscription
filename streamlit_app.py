import os
import wave
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template
from pydub import AudioSegment
import speech_recognition as sr
from langdetect import detect
from textblob import TextBlob

app = Flask(__name__)

class AudioTranscriptionService:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert_to_wav(self, file_path):
        """Convert audio file to WAV format if necessary."""
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension != ".wav":
            audio = AudioSegment.from_file(file_path)
            wav_file_path = file_path.replace(file_extension, ".wav")
            audio.export(wav_file_path, format="wav")
            return wav_file_path
        return file_path

    def split_audio(self, file_path, chunk_duration_ms=60000):
        """Split large audio files into smaller chunks."""
        audio = AudioSegment.from_file(file_path)
        chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]
        chunk_paths = []
        for idx, chunk in enumerate(chunks):
            chunk_path = f"{file_path}_chunk_{idx}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
        return chunk_paths

    def transcribe_chunk(self, chunk_path, language="en-US"):
        """Transcribe a single chunk of audio."""
        try:
            with sr.AudioFile(chunk_path) as source:
                audio_data = self.recognizer.record(source)
                return self.recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            return "[Unintelligible audio]"
        except sr.RequestError as e:
            return f"Error: {e}"

    def transcribe_audio(self, file_path, language="en-US"):
        """Transcribe audio file with support for large files."""
        file_path = self.convert_to_wav(file_path)
        chunks = self.split_audio(file_path)
        transcription = []

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda chunk: self.transcribe_chunk(chunk, language), chunks)
            transcription.extend(results)

        return transcription

    def detect_language(self, transcription):
        """Detect the language of the transcription."""
        try:
            detected_language = detect(" ".join(transcription))
            return detected_language
        except Exception as e:
            return "Error detecting language."

    def analyze_sentiment(self, transcription):
        """Perform sentiment analysis on the transcription."""
        text = " ".join(transcription)
        sentiment = TextBlob(text).sentiment
        return {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    language = request.form.get("language", "en-US")
    service = AudioTranscriptionService()
    transcription = service.transcribe_audio(file_path, language)
    detected_language = service.detect_language(transcription)
    sentiment_analysis = service.analyze_sentiment(transcription)

    response = {
        "transcription": transcription,
        "detected_language": detected_language,
        "sentiment_analysis": sentiment_analysis,
    }

    return jsonify(response)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
