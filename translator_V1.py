import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import soundfile as sf
import numpy as np
import whisper
import functools
import traceback 

# Global cache for models
MODEL_CACHE = {
    'translation': {},
    'whisper': {}  
}

def cache_model(cache_dict, key_func=None):
    """
    Decorator to cache model loading with flexible key generation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = key_func(*args, **kwargs) if key_func else None
            
            # Check cache first
            if key and key in cache_dict:
                logging.info(f"Using cached model for key: {key}")
                return cache_dict[key]
            
            # Load model
            model = func(*args, **kwargs)
            
            # Cache if key provided
            if key:
                cache_dict[key] = model
            
            return model
        return wrapper
    return decorator

@cache_model(MODEL_CACHE['whisper'], key_func=lambda model_name='base': model_name)
def load_model(model_name='base'):
    """Load Whisper model with caching"""
    logging.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    return model

def transcribe(audio, model):
    # Validate input
    if audio is None:
        logging.error("Audio input is None")
        return None
    
    # Ensure minimum audio length
    min_length = 8000  # 0.5 seconds at 16kHz
    if len(audio) < min_length:
        logging.warning(f"Audio too short. Padding to {min_length} samples")
        padding = np.zeros(min_length - len(audio))
        audio = np.concatenate([audio, padding])

    # Ensure float32 type
    audio = np.array(audio, dtype=np.float32)

    # Perform transcription
    try:
        result = model.transcribe(audio, fp16=False)
        transcribed_text = result.get('text', '').strip()
        
        if not transcribed_text:
            logging.error("Transcription returned empty text")
            return None
        
        return transcribed_text
    
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        logging.error(traceback.format_exc())
        return None

@cache_model(MODEL_CACHE['translation'], key_func=lambda source_lang, target_lang: f"{source_lang}-{target_lang}")
def load_translation_model(source_lang, target_lang):
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        logging.info(f"Attempting to load model for {source_lang}-{target_lang}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        return tokenizer, model
    except Exception as e:
        logging.error(f"Model loading error: {e}")
        logging.error(traceback.format_exc())
        raise ValueError(f"Failed to load translation model: {e}")

# Mapping for common language codes
LANG_MAP = {
    'en': 'eng_Latn',
    'es': 'spa_Latn',
    'fr': 'fra_Latn',
    'el': 'gre_Latn',
    'yor': 'yor_Latn',
    # Add more language codes as needed
}

def translate_text(text, source_lang='en', target_lang='es'):
    """Enhanced translation with improved generation parameters"""
    # Validate input
    if not text or not text.strip():
        return ""
    
    try:
        # Convert language codes using mapping
        source_lang = LANG_MAP.get(source_lang, source_lang)
        target_lang = LANG_MAP.get(target_lang, target_lang)
        
        # Load translation model
        tokenizer, model = load_translation_model(source_lang, target_lang)
        
        # Prepare inputs
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate translation with enhanced parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_length=512,  # Maximum output length
                num_beams=8,     # Increased beam width for better exploration
                length_penalty=1.2,  # Slight length penalty to favor slightly longer translations
                no_repeat_ngram_size=2,  # Prevent repetition of 2-gram sequences
                early_stopping=True,
                do_sample=True,  # Add some randomness to beam search
                top_k=50,        # Top-k sampling
                top_p=0.95,      # Nucleus sampling
                temperature=0.7  # Control randomness of sampling
            )
        
        # Decode translated text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Translated text: {translated_text}")
        
        return translated_text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        logging.error(traceback.format_exc())
        raise ValueError(f"Translation failed: {e}")

def text_to_speech(translated_text, language, output_file):
    tts = gTTS(translated_text, lang=language)
    tts.save(output_file)
    logging.info(f"Translated audio saved as '{output_file}'")

def process_audio(audio_path, to_language, output_file):
    # Load the Whisper model
    model = load_model()

    # Load an audio file
    audio, sample_rate = sf.read(audio_path)

    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
            np.arange(len(audio)),
            audio
        )

    # Perform transcription using Whisper
    transcription = transcribe(audio, model)
    if transcription is None:
        raise ValueError("Transcription failed. Please check the input audio.")

    logging.info(f"Transcription: {transcription}")

    # Translate the transcription using MarianMT
    translated_text = translate_text(transcription, source_lang='en', target_lang=to_language)
    logging.info(f"Translated Transcription: {translated_text}")

    # Convert translated text to audio
    text_to_speech(translated_text, to_language, output_file)
