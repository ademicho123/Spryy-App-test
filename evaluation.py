import logging
from typing import List, Dict
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from translator_V1 import translate_text as v1_translate
from translator_MT import translate_text as mt_translate

class TranslationEvaluator:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        self.logger = logging.getLogger(__name__)
    
    def load_test_data(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load test data from a JSON or CSV file
        
        :param file_path: Path to test data file
        :return: List of translation test cases
        """
        import json
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            return []
    
    def evaluate_translation(self, 
                              source_texts: List[str], 
                              reference_translations: List[str], 
                              translator_func):
        """
        Evaluate translation quality using multiple metrics
        
        :param source_texts: List of source texts
        :param reference_translations: List of reference (ground truth) translations
        :param translator_func: Translation function to evaluate
        :return: Dictionary of evaluation metrics
        """
        # Translate texts
        translated_texts = [translator_func(text) for text in source_texts]
        
        # Tokenize for BLEU score
        reference_tokens = [[word_tokenize(ref.lower())] for ref in reference_translations]
        candidate_tokens = [word_tokenize(trans.lower()) for trans in translated_texts]
        
        # Calculate BLEU score
        bleu_score = corpus_bleu(reference_tokens, candidate_tokens)
        
        # Calculate other simple metrics
        exact_matches = sum(1 for t, r in zip(translated_texts, reference_translations) if t.strip() == r.strip())
        
        return {
            'bleu_score': bleu_score,
            'exact_matches': exact_matches,
            'total_texts': len(source_texts),
            'match_percentage': (exact_matches / len(source_texts)) * 100
        }
    
    def compare_translators(self, test_data_path: str):
        """
        Compare performance of V1 and MarianMT translators
        
        :param test_data_path: Path to test data file
        :return: Comparison results
        """
        test_data = self.load_test_data(test_data_path)
        
        source_texts = [item['source'] for item in test_data]
        reference_translations = [item['reference'] for item in test_data]
        
        v1_results = self.evaluate_translation(source_texts, reference_translations, v1_translate)
        mt_results = self.evaluate_translation(source_texts, reference_translations, mt_translate)
        
        return {
            'V1_Translator': v1_results,
            'MarianMT_Translator': mt_results
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = TranslationEvaluator()
    
    # Assume you have a test_translations.json with source, reference translations
    results = evaluator.compare_translators('test_translations.json')
    
    print("Translation Evaluation Results:")
    for translator, metrics in results.items():
        print(f"\n{translator}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")