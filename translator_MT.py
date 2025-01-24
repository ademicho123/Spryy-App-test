import torch
import logging
from transformers import MarianMTModel, MarianTokenizer

class MarianTranslator:
    def __init__(self, source_lang='en', target_lang='es'):
        """
        Initialize MarianMT translator with robust error handling
        
        :param source_lang: Source language code (default: English)
        :param target_lang: Target language code (default: Spanish)
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            # Construct model name
            model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
            
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
        except ImportError as e:
            self.logger.error("Missing library. Install with: pip install sentencepiece transformers")
            raise
        except Exception as e:
            self.logger.error(f"Error loading translation model: {e}")
            raise
    
    def translate(self, text):
        """
        Translate text using MarianMT
        
        :param text: Input text to translate
        :return: Translated text
        """
        try:
            # Prepare the input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Decode the translation
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_text
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return None
    
    @classmethod
    def supported_languages(cls):
        """
        List some supported language pairs
        """
        return [
            'en-es', 'en-fr', 'en-de', 
            'es-en', 'fr-en', 'de-en'
        ]

def translate_text(text, source_lang='en', target_lang='es'):
    """
    Convenience function for text translation
    
    :param text: Text to translate
    :param source_lang: Source language code
    :param target_lang: Target language code
    :return: Translated text
    """
    translator = MarianTranslator(source_lang, target_lang)
    return translator.translate(text)