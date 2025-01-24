from flask import Flask, request, jsonify, send_file
import soundfile as sf
import numpy as np
import tempfile
import os

from translator_MT import MarianTranslator
from translator_V1 import process_audio, transcribe, load_model

app = Flask(__name__)

@app.route('/translate/text', methods=['POST'])
def text_translation():
    """Flexible text translation API"""
    # Try JSON first
    if request.is_json:
        data = request.get_json()
    else:
        # Try form data if JSON fails
        data = request.form

    text = data.get('text')
    source_language = data.get('source_language', 'en')
    target_language = data.get('target_language', 'es')

    if not text:
        return jsonify({'error': 'Missing text'}), 400
    if not target_language:
        return jsonify({'error': 'Missing target language'}), 400

    try:
        translator = MarianTranslator(source_language, target_language)
        translated_text = translator.translate(text)
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate/speech', methods=['POST'])
def speech_translation():
    """Speech translation API"""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio_file']
    source_language = request.form.get('source_language', 'en')
    target_language = request.form.get('target_language', 'es')
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as input_temp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as output_temp:
            audio_file.save(input_temp.name)
            
            # Process audio translation
            process_audio(input_temp.name, target_language, output_temp.name)
            
            # Return the translated audio file
            return send_file(output_temp.name, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary files
        try:
            os.unlink(input_temp.name)
            os.unlink(output_temp.name)
        except:
            pass

@app.route('/transcribe/speech', methods=['POST'])
def speech_to_text():
    """Speech-to-Text Transcription API"""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio_file']
    language = request.form.get('language', 'en')
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            
            # Load model
            processor, model = load_model()
            
            # Read audio file
            audio, sample_rate = sf.read(temp_file.name)
            
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
            
            # Transcribe
            transcription = transcribe(audio, processor, model)
            
            return jsonify({
                'transcription': transcription,
                'language': language
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file.name)
        except:
            pass

@app.route('/supported_languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported translation languages"""
    return jsonify({
        'supported_languages': MarianTranslator.supported_languages()
    })

if __name__ == '__main__':
    app.run(debug=True)