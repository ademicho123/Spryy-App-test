import os
import azure.cognitiveservices.speech as speechsdk
import requests
import logging
from flask_socketio import emit
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Get API keys and other details from environment variables
speech_subscription_key = os.getenv('AZURE_SPEECH_KEY')
speech_region = os.getenv('AZURE_REGION')
translator_subscription_key = os.getenv('AZURE_TRANSLATOR_KEY')
translator_endpoint = os.getenv('AZURE_TRANSLATOR_ENDPOINT', "https://api.cognitive.microsofttranslator.com/")

# Store speaker profiles and languages
speaker_profiles = {}
user_languages = {}

def init_speech_recognizer(language):
    try:
        logging.info(f"Initializing speech recognizer with subscription key: {speech_subscription_key} and region: {speech_region}")
        speech_config = speechsdk.SpeechConfig(subscription=speech_subscription_key, region=speech_region)
        speech_config.speech_recognition_language = language
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        logging.info(f"Initialized speech recognizer with language: {language}")
        return recognizer
    except Exception as e:
        logging.error(f"Failed to initialize speech recognizer: {e}")
        raise

def translate_text(text, to_language):
    path = '/translate'
    constructed_url = translator_endpoint.rstrip('/') + path

    params = {
        'api-version': '3.0',
        'to': to_language
    }

    headers = {
        'Ocp-Apim-Subscription-Key': translator_subscription_key,
        'Ocp-Apim-Subscription-Region': speech_region,
        'Content-type': 'application/json'
    }

    body = [{
        'text': text
    }]

    try:
        response = requests.post(constructed_url, params=params, headers=headers, json=body)
        response.raise_for_status()
        translations = response.json()
        translated_text = translations[0]['translations'][0]['text']
        logging.info(f"Translated text: {translated_text}")
        return translated_text
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error during translation: {e}")
        logging.error(f"Response content: {e.response.text}")
        raise
    except Exception as e:
        logging.error(f"Error during translation: {e}")
        raise

def real_time_translate_audio(from_language, to_language, user_id):
    try:
        recognizer = init_speech_recognizer(from_language)
        
        def recognized_cb(event):
            try:
                text = event.result.text
                logging.info(f"Recognized text: {text}")

                translated_text = translate_text(text, to_language)
                logging.info(f"Translated text: {translated_text}")

                emit('translated_text', translated_text, room=user_id)
                
            except Exception as e:
                logging.error(f"Error during recognized callback: {e}")

        recognizer.recognized.connect(recognized_cb)
        recognizer.start_continuous_recognition()

        logging.info("Started real-time audio translation. Press Ctrl+C to stop.")
        while True:
            pass  # Keep the process running for continuous recognition

    except Exception as e:
        logging.error(f"Error in real-time translation: {e}")
        raise
