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

def normalize_text(text: str) -> str:
    """
    Advanced text normalization for more accurate comparison
    - Convert to lowercase
    - Normalize Unicode characters
    - Remove diacritical marks
    - Expand common contractions
    - Normalize punctuation and whitespace
    """
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove diacritical marks
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Lowercase
    text = text.lower()
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Normalize punctuation
    text = re.sub(r'[''""]', "'", text)
    text = re.sub(r'[—–]', '-', text)
    
    # Remove extra whitespaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def expand_contractions(text: str) -> str:
    """Expand common English contractions"""
    contractions = {
        "n't": " not",
        "'m": " am",
        "'s": " is",
        "'re": " are",
        "'ll": " will",
        "'ve": " have",
        "'d": " would"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    return text

def calculate_enhanced_similarity(ref_text: str, trans_text: str) -> float:
    """
    Enhanced similarity calculation considering word order and semantic similarity
    """
    normalized_ref = normalize_text(ref_text)
    normalized_trans = normalize_text(trans_text)
    
    # Use SequenceMatcher for base similarity
    base_similarity = difflib.SequenceMatcher(None, normalized_ref, normalized_trans).ratio()
    
    # Additional token-based similarity
    ref_tokens = word_tokenize(normalized_ref)
    trans_tokens = word_tokenize(normalized_trans)
    
    token_overlap = len(set(ref_tokens) & set(trans_tokens)) / max(len(ref_tokens), len(trans_tokens), 1)
    
    # Combine methods
    return (base_similarity + token_overlap) / 2

def evaluate_translation(source_text: str, 
                         reference_translation: str, 
                         translated_text: str):
    """Enhanced translation quality assessment with advanced scoring"""
    # Advanced normalization
    source_text = normalize_text(str(source_text).strip())
    reference_translation = normalize_text(str(reference_translation).strip())
    translated_text = normalize_text(str(translated_text).strip())

    reference_tokens = word_tokenize(reference_translation)
    candidate_tokens = word_tokenize(translated_text)
    
    try:
        # More nuanced BLEU calculation with adjusted weights
        bleu_score = sentence_bleu(
            [reference_tokens], 
            candidate_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25)  # Balanced n-gram weights
        )
    except Exception as e:
        logging.error(f"BLEU score error: {e}")
        bleu_score = 0.0
    
    similarity_ratio = calculate_enhanced_similarity(reference_translation, translated_text)
    
    return {
        'bleu_score': bleu_score,
        'similarity_ratio': similarity_ratio,
        'normalized_reference': reference_translation,
        'normalized_translation': translated_text
    }
    
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