import logging
from typing import List, Dict, Callable
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from jiwer import wer as jiwer_wer
from nltk.metrics import edit_distance
from nltk.translate.meteor_score import meteor_score
import difflib
import unicodedata
import re

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def normalize_text(text):
    """Advanced text normalization"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = ''.join(
        char for char in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(char)
    )
    
    # Normalize specific characters and punctuation
    text = re.sub(r'[\'"`''""()]', '', text)  # Remove various quote types
    text = re.sub(r'[?!.,;:]', ' ', text)    # Replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    
    return text

def compute_advanced_translation_score(reference, translation):
    """
    Compute a comprehensive translation quality score
    
    Combines multiple metrics:
    - METEOR score (more semantically aware)
    - Normalized token-level similarity
    - Sequence matching
    """
    # Normalize texts
    norm_ref = normalize_text(reference)
    norm_trans = normalize_text(translation)
    
    # Tokenize normalized texts
    ref_tokens = word_tokenize(norm_ref)
    trans_tokens = word_tokenize(norm_trans)
    
    # METEOR score (more semantically sensitive than BLEU)
    try:
        meteor = meteor_score([ref_tokens], trans_tokens)
    except Exception:
        meteor = 0.0
    
    # Sequence similarity
    sequence_sim = difflib.SequenceMatcher(None, norm_ref, norm_trans).ratio()
    
    # Token overlap
    unique_ref = set(ref_tokens)
    unique_trans = set(trans_tokens)
    token_overlap = len(unique_ref & unique_trans) / max(len(unique_ref), len(unique_trans), 1)
    
    # Classic BLEU score with more forgiving weights
    try:
        bleu = sentence_bleu(
            [ref_tokens], 
            trans_tokens, 
            weights=(0.5, 0.3, 0.2, 0)  # Less strict n-gram weights
        )
    except Exception:
        bleu = 0.0
    
    # Combine metrics with weighted average
    combined_score = (
        0.4 * meteor +  # METEOR score
        0.3 * sequence_sim +  # Sequence similarity
        0.2 * token_overlap +  # Token overlap
        0.1 * bleu  # BLEU score
    )
    
    return {
        'bleu_score': bleu,
        'meteor_score': meteor,
        'semantic_similarity': combined_score,
        'normalized_reference': norm_ref,
        'normalized_translation': norm_trans
    }

def evaluate_translation(source_text: str, 
                         reference_translation: str, 
                         translated_text: str):
    """Comprehensive translation quality assessment"""
    return compute_advanced_translation_score(
        reference_translation, 
        translated_text
    )
    
def evaluate_speech_to_text(transcription: str, 
                             reference_transcription: str):
    """
    Evaluate speech-to-text quality using multiple metrics with case-insensitive comparison
    
    :param transcription: Transcription
    :param reference_transcription: Reference transcription
    :return: Dictionary of evaluation metrics
    """
    # Convert both texts to lowercase for comparison
    transcription_lower = transcription.strip().lower()
    reference_lower = reference_transcription.strip().lower()
    
    # Calculate WER score
    wer_score = jiwer_wer(reference_lower, transcription_lower)
    
    # Calculate exact match (case-insensitive)
    exact_match = 1 if transcription_lower == reference_lower else 0
    
    return {
        'wer_score': wer_score,
        'exact_match': exact_match
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the evaluation functions
    source_text = "What do you do in your free time?!"
    reference_translation = "Qu'est-ce que vous aimez faire pendant votre temps libre ?"
    
    # For text translation
    from translator_V1 import translate_text
    translated_text = translate_text(source_text)
    
    text_translation_results = evaluate_translation(
        source_text, 
        reference_translation, 
        translated_text
    )
    logging.info("Text Translation Evaluation Results:")
    logging.info(text_translation_results)