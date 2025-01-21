import logging
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from deep_translator import GoogleTranslator
from gtts import gTTS
import soundfile as sf
import numpy as np

def load_model(model_name='facebook/wav2vec2-base-960h'):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def transcribe(audio, processor, model):
    # Check if audio is too short (less than 0.5 seconds)
    if len(audio) < 8000:  # Assuming 16kHz sample rate
        # Pad the audio to 0.5 seconds
        padding = np.zeros(8000 - len(audio))
        audio = np.concatenate([audio, padding])
    
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    
    # Get the attention mask
    attention_mask = torch.ones(input_values.shape, dtype=torch.long, device=input_values.device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription
    
def translate_text(text, to_language):
    translator = GoogleTranslator(source='auto', target=to_language)
    translated_text = translator.translate(text)
    logging.info(f"Translated text: {translated_text}")
    return translated_text

def text_to_speech(translated_text, language, output_file):
    tts = gTTS(translated_text, lang=language)
    tts.save(output_file)
    logging.info(f"Translated audio saved as '{output_file}'")

def process_audio(audio_path, to_language, output_file):
    # Load the model and processor
    processor, model = load_model()

    # Load an audio file
    audio, sample_rate = sf.read(audio_path)

    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        audio = np.interp(np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)), np.arange(len(audio)), audio)

    # Perform transcription
    result = transcribe(audio, processor, model)
    logging.info(f"Transcription: {result}")

    # Translate the transcription
    translated_result = translate_text(result, to_language)
    logging.info(f"Translated Transcription: {translated_result}")

    # Convert translated text to audio
    text_to_speech(translated_result, to_language, output_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audio_path = "Spryy-App/spryy-app-web/harvard.wav"
    to_language = "es"  # Change to your desired language code
    output_file = "translated_audio.mp3"
    process_audio(audio_path, to_language, output_file)
