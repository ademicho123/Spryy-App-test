import logging
from flask import Flask, request, jsonify, send_file
import soundfile as sf
import numpy as np
import tempfile
import os
import traceback

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    reference_translation = data.get('reference_translation')

    if not text:
        return jsonify({'error': 'Missing text'}), 400
    if not target_language:
        return jsonify({'error': 'Missing target language'}), 400

    try:
        from translator_V1 import translate_text
        translated_text = translate_text(text, source_language, target_language)
        
        if not translated_text:
            return jsonify({'error': 'Translation failed'}), 500
        
        response = {
            'translated_text': translated_text
        }
        
        if reference_translation:
            from evaluation import evaluate_translation
            evaluation_results = evaluate_translation(
                source_text=text, 
                reference_translation=reference_translation, 
                translated_text=translated_text
            )
            response['evaluation_results'] = evaluation_results
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe/speech', methods=['POST'])
def speech_to_text_and_translation():
    audio_file = request.files.get('audio_file')
    source_language = request.form.get('source_language', 'en')
    target_language = request.form.get('target_language', 'es')
    reference_transcription = request.form.get('reference_transcription')
    reference_translation = request.form.get('reference_translation')

    if not audio_file:
        return jsonify({'error': 'No audio file uploaded'}), 400

    temp_file = None
    try:
        # Create temp file with explicit .wav extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_file.name)
        logger.debug(f"Saved temp audio file: {temp_file.name}")

        # Verify file was saved correctly
        if os.path.getsize(temp_file.name) == 0:
            raise ValueError("Uploaded audio file is empty")

        from translator_V1 import load_model, transcribe, translate_text
        from evaluation import evaluate_speech_to_text, evaluate_translation

        # Step 1: Transcribe Audio
        model = load_model()
        logger.debug("Whisper model loaded successfully")

        # Read audio file with detailed error handling
        try:
            audio, sample_rate = sf.read(temp_file.name)
            logger.debug(f"Audio read - Shape: {audio.shape}, Sample Rate: {sample_rate}")
        except Exception as e:
            logger.error(f"Audio read error: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Audio read failed: {str(e)}'}), 400

        # Handle multi-channel audio
        audio = audio.mean(axis=1) if len(audio.shape) > 1 else audio
        logger.debug(f"Audio normalized. New shape: {audio.shape}")

        # Resample if needed
        audio = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / sample_rate)),
            np.arange(len(audio)),
            audio
        ) if sample_rate != 16000 else audio
        logger.debug(f"Audio resampled. Final shape: {audio.shape}")

        # Transcription with detailed logging
        transcription = transcribe(audio, model)
        if not transcription:
            return jsonify({'error': 'Transcription failed'}), 500
        logger.info(f"Transcription: {transcription}")

        # Step 2: Translate Transcription
        translated_text = translate_text(transcription, source_language, target_language)
        logger.info(f"Translation: {translated_text}")

        # Step 3-4: Evaluation (if references provided)
        response = {
            'transcription': transcription,
            'translation': translated_text,
        }

        if reference_transcription:
            response['transcription_evaluation'] = evaluate_speech_to_text(
                transcription, reference_transcription
            )

        if reference_translation:
            response['translation_evaluation'] = evaluate_translation(
                source_text=transcription,
                reference_translation=reference_translation,
                translated_text=translated_text
            )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Comprehensive error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

    finally:
        # Ensure temp file is always deleted
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.debug("Temporary file deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")
                
if __name__ == '__main__':
    app.run(debug=True)
