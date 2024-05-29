import os
import time
import yt_dlp
import requests
import traceback
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# YouTube to WAV
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

def download_from_url(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(result).rsplit('.', 1)[0] + '.wav'

# Diarization and Transcription
def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs, transcribed_text):
    try:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcribed_text.append(f"Speaker {evt.result.speaker_id}: {evt.result.text}\n")
    except Exception as e:
        traceback.print_exc()
        print("Error in transcription callback:", e)

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

def recognize_from_file(audio_file):
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["hi-IN", "mr-IN", "te-IN"])
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, auto_detect_source_language_config=auto_detect_source_language_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once()
    auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(result)
    detected_language = auto_detect_source_language_result.language
    print("Detected language:", detected_language)

    # Check if the language was detected successfully
    if not detected_language:
        print("Language detection failed. Using default language: en-US")
        detected_language = "en-US"  # Fallback to English

    speech_config.speech_recognition_language = detected_language
    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    transcribed_text = []

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    conversation_transcriber.transcribed.connect(lambda evt: conversation_transcriber_transcribed_cb(evt, transcribed_text))
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    conversation_transcriber.start_transcribing_async()

    while not transcribing_stop:
        time.sleep(.5)

    conversation_transcriber.stop_transcribing_async()
    print("Transcription completed successfully.")

    return ''.join(transcribed_text)

# Translation
def translate_text(text):
    chunks = text.split('\n')
    translated_chunks = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        print(f"Translating chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        prompt = "Translate the following text to English:"
        output = get_openai_response(prompt, chunk)
        if output:
            print(f"Translation for chunk {i+1}/{len(chunks)} completed: {output[:50]}...")
            translated_chunks.append(output)
        else:
            print(f"Failed to get translation for chunk {i+1}/{len(chunks)}.")
    
    return '\n'.join(translated_chunks)

def get_openai_response(prompt, input_text):
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_base_url = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    api_version = '2022-12-01'
    deployment_id = 'ver01'
    api_url = f"{api_base_url}/openai/deployments/{deployment_id}/completions?api-version={api_version}"
    
    payload = {
        "prompt": f"{prompt}\n\nInput: {input_text}\n\nOutput:",
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.95,
        "n": 1,
        "stop": ["\n"]
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    try:
        print(f"Request Payload: {payload}")
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        completion = response.json()
        if "choices" in completion and len(completion["choices"]) > 0:
            return completion["choices"][0]["text"].strip()
        else:
            print("Unexpected response format or empty response:", completion)
            return None
    except requests.RequestException as e:
        print("Request failed:", e)
        if e.response is not None:
            print("Response status code:", e.response.status_code)
            print("Response content:", e.response.text)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    choice = request.form['choice']
    
    if choice == '1':
        url = request.form['url']
        audio_file = download_from_url(url)
    elif choice == '2':
        audio_file = request.files['audio_file']
        audio_file.save(audio_file.filename)
        audio_file = audio_file.filename
    else:
        return jsonify({'error': 'Invalid choice.'}), 400

    transcribed_text = recognize_from_file(audio_file)
    
    if transcribed_text:
        translated_text = translate_text(transcribed_text)
        
        if translated_text:
            return jsonify({'transcribed_text': transcribed_text, 'translated_text': translated_text})
        else:
            return jsonify({'error': 'Translation failed.'}), 500
    else:
        return jsonify({'error': 'Transcription failed.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
