import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
import numpy as np
import librosa
import logging
from scipy.io.wavfile import write as write_wav
from vits import utils, commons
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence
from speechbrain.pretrained import EncoderClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_asr_model(model_name='facebook/wav2vec2-base-960h'):
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def load_translation_model(model_name='Helsinki-NLP/opus-mt-en-es'):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def load_vits_model(model_path, config_path):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    return net_g, hps

def load_emotion_classifier():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
    return classifier

def transcribe(audio, processor, model):
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

def extract_speaker_embedding(audio, vits_model, hps):
    audio = torch.FloatTensor(audio).unsqueeze(0)
    y = audio.to(torch.float)
    spec = utils.spectrogram_torch(y, hps.data.filter_length,
                                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                center=False)
    spec_lengths = torch.LongTensor([spec.size(-1)])
    sid = torch.LongTensor([0])  # Speaker ID, we use 0 as we have only one speaker
    
    with torch.no_grad():
        z, _, _, _ = vits_model.enc_p(spec, spec_lengths, sid=sid)
    return z.squeeze().mean(dim=0)

def recognize_emotion(audio, emotion_classifier):
    waveform = torch.tensor(audio).unsqueeze(0)
    emotion_logits = emotion_classifier.classify_batch(waveform)
    emotion_probs = torch.softmax(emotion_logits, dim=-1)
    emotion = emotion_classifier.hparams.label_encoder.decode_ndim(torch.argmax(emotion_probs, dim=-1))
    return emotion[0]

def synthesize_speech(text, speaker_embedding, emotion, vits_model, hps):
    text = text.strip()
    stn_tst = text_to_sequence(text, hps.symbols)
    with torch.no_grad():
        x_tst = torch.LongTensor(stn_tst).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([len(stn_tst)])
        
        # Adjust noise_scale and noise_scale_w based on emotion
        noise_scale = 0.667
        noise_scale_w = 0.8
        if emotion in ['angry', 'fearful']:
            noise_scale = 0.8
            noise_scale_w = 1.0
        elif emotion in ['sad', 'disgusted']:
            noise_scale = 0.5
            noise_scale_w = 0.6
        elif emotion in ['happy', 'surprised']:
            noise_scale = 0.7
            noise_scale_w = 0.9
        
        audio = vits_model.infer(x_tst, x_tst_lengths, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=1, sid=speaker_embedding)[0][0,0].data.cpu().float().numpy()
    return audio

def voice_to_voice_translation(input_audio_path, output_audio_path, vits_model_path, vits_config_path, src_lang='en', tgt_lang='es'):
    # Load models
    asr_processor, asr_model = load_asr_model()
    trans_tokenizer, trans_model = load_translation_model(f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}')
    vits_model, vits_hps = load_vits_model(vits_model_path, vits_config_path)
    emotion_classifier = load_emotion_classifier()

    try:
        # Load and preprocess audio
        audio, sample_rate = librosa.load(input_audio_path, sr=16000)
        
        # Extract speaker embedding
        speaker_embedding = extract_speaker_embedding(audio, vits_model, vits_hps)
        
        # Recognize emotion
        emotion = recognize_emotion(audio, emotion_classifier)
        logging.info(f"Recognized emotion: {emotion}")
        
        # Transcribe
        transcription = transcribe(audio, asr_processor, asr_model)
        logging.info(f"Transcribed text: {transcription}")

        # Translate
        translated_text = translate(transcription, trans_tokenizer, trans_model)
        logging.info(f"Translated text: {translated_text}")

        # Synthesize speech
        output_audio = synthesize_speech(translated_text, speaker_embedding, emotion, vits_model, vits_hps)
        
        # Save output audio
        write_wav(output_audio_path, vits_hps.data.sampling_rate, output_audio)
        logging.info(f"Generated speech saved to {output_audio_path}")

        return transcription, translated_text, emotion, output_audio_path

    except Exception as e:
        logging.error(f"Error in voice-to-voice translation: {e}")
        raise

if __name__ == "__main__":
    input_audio = "path/to/input/audio.wav"
    output_audio = "path/to/output/audio.wav"
    vits_model_path = "path/to/vits/model.pth"
    vits_config_path = "path/to/vits/config.json"
    src_language = "en"
    target_language = "es"

    transcription, translation, emotion, output_path = voice_to_voice_translation(
        input_audio, output_audio, vits_model_path, vits_config_path, 
        src_lang=src_language, tgt_lang=target_language
    )

    print(f"Transcription: {transcription}")
    print(f"Translation: {translation}")
    print(f"Detected Emotion: {emotion}")
    print(f"Output audio saved to: {output_path}")