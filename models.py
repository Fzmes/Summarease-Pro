import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    T5ForConditionalGeneration,
    T5Tokenizer,
    MarianMTModel,
    MarianTokenizer
)
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedMultilingualModels:
    """Gestionnaire de modèles multilingues étendu avec support arabe et scraping web"""

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models_loaded = False

        # Mapping étendu des langues avec codes
        self.language_codes = {
            "français": "fr", "fr": "fr", "french": "fr",
            "anglais": "en", "en": "en", "english": "en",
            "espagnol": "es", "es": "es", "spanish": "es",
            "allemand": "de", "de": "de", "german": "de",
            "italien": "it", "it": "it", "italian": "it",
            "portugais": "pt", "pt": "pt", "portuguese": "pt",
            "néerlandais": "nl", "nl": "nl", "dutch": "nl",
            "russe": "ru", "ru": "ru", "russian": "ru",
            "chinois": "zh", "zh": "zh", "chinese": "zh",
            "japonais": "ja", "ja": "ja", "japanese": "ja",
            "arabe": "ar", "ar": "ar", "arabic": "ar",
            "coréen": "ko", "ko": "ko", "korean": "ko",
            "hindi": "hi", "hi": "hi", "hindi": "hi"
        }

        self._load_models()

    def _setup_device(self, device: str) -> torch.device:
        """Configure le device (GPU/CPU)"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _load_models(self):
        """Charge tous les modèles avec support étendu"""
        try:
            logger.info("🚀 Chargement des modèles multilingues étendus...")

            # Modèles de résumé
            self.summary_models = self._load_summary_models()

            # Modèles de traduction avec support arabe
            self.translation_models = self._load_translation_models()

            self.models_loaded = True
            logger.info("✅ Modèles étendus chargés avec succès!")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
            self.models_loaded = False

    def _load_summary_models(self) -> Dict:
        """Charge les modèles de résumé"""
        models = {}

        # Modèle français
        try:
            models["fr"] = pipeline(
                "summarization",
                model="moussaKam/barthez-orangesum-abstract",
                tokenizer="moussaKam/barthez-orangesum-abstract",
                device=self.device
            )
            logger.info("✅ Modèle de résumé français chargé")
        except Exception as e:
            logger.warning(f"Modèle français non disponible: {e}")

        # Modèle anglais
        try:
            models["en"] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                device=self.device
            )
            logger.info("✅ Modèle de résumé anglais chargé")
        except Exception as e:
            logger.warning(f"Modèle anglais non disponible: {e}")

        # Modèle multilingue pour autres langues
        try:
            models["multilingual"] = pipeline(
                "summarization",
                model="google/mt5-small",
                tokenizer="google/mt5-small",
                device=self.device
            )
            logger.info("✅ Modèle de résumé multilingue chargé")
        except Exception as e:
            logger.warning(f"Modèle multilingue non disponible: {e}")

        return models

    def _load_translation_models(self) -> Dict:
        """Charge les modèles de traduction avec support arabe"""
        models = {}

        # Modèle M2M100 (supporte 100 langues dont l'arabe)
        try:
            models["m2m100"] = {
                "model": AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M"),
                "tokenizer": AutoTokenizer.from_pretrained("facebook/m2m100_418M"),
                "type": "m2m100"
            }
            logger.info("✅ Modèle M2M100 chargé (supporte arabe)")
        except Exception as e:
            logger.warning(f"Modèle M2M100 non disponible: {e}")

        # Modèle MBART-50 (supporte 50 langues dont l'arabe)
        try:
            models["mbart50"] = {
                "model": MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt"),
                "tokenizer": MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt"),
                "type": "mbart50"
            }
            logger.info("✅ Modèle MBART-50 chargé (supporte arabe)")
        except Exception as e:
            logger.warning(f"Modèle MBART-50 non disponible: {e}")

        # Chargement des modèles MarianMT disponibles
        available_marian_models = {
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "fr-es": "Helsinki-NLP/opus-mt-fr-es",
            "es-fr": "Helsinki-NLP/opus-mt-es-fr",
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "en-de": "Helsinki-NLP/opus-mt-en-de",
            "de-en": "Helsinki-NLP/opus-mt-de-en",
            "en-it": "Helsinki-NLP/opus-mt-en-it",
            "it-en": "Helsinki-NLP/opus-mt-it-en",
            "en-ar": "Helsinki-NLP/opus-mt-en-ar",  # Anglais -> Arabe
            "ar-en": "Helsinki-NLP/opus-mt-ar-en",  # Arabe -> Anglais
            "fr-de": "Helsinki-NLP/opus-mt-fr-de",  # Français -> Allemand
            "de-fr": "Helsinki-NLP/opus-mt-de-fr",  # Allemand -> Français
            "es-de": "Helsinki-NLP/opus-mt-es-de",  # Espagnol -> Allemand
            "de-es": "Helsinki-NLP/opus-mt-de-es",  # Allemand -> Espagnol
        }

        for pair, model_name in available_marian_models.items():
            try:
                models[f"marian_{pair}"] = {
                    "model": MarianMTModel.from_pretrained(model_name),
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "type": "marian"
                }
                logger.info(f"✅ Modèle MarianMT {pair} chargé")
            except Exception as e:
                logger.warning(f"Modèle MarianMT {pair} non disponible: {e}")

        return models

    def scrape_web_content(self, url):
        """Extrait le contenu d'une page web de manière robuste"""
        try:
            # Vérifier que l'URL est valide
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
            }
            
            # Faire la requête avec timeout et vérification du statut
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()  # Lève une exception pour les codes 4xx/5xx
            
            # Vérifier le type de contenu
            content_type = response.headers.get('content-type', '')
            if 'text/html' not in content_type:
                return f"Erreur: Le contenu n'est pas HTML (type: {content_type})"
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Supprimer les éléments non désirés de manière plus complète
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "menu"]):
                element.decompose()
            
            # Stratégie d'extraction multiple
            text_content = []
            
            # 1. Essayer les balises sémantiques HTML5 d'abord
            main_content = soup.find(['article', 'main', '[role="main"]'])
            if main_content:
                paragraphs = main_content.find_all(['p', 'div'])
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text.split()) > 5:  # Au moins 5 mots
                        text_content.append(text)
            
            # 2. Si pas de contenu principal trouvé, chercher plus largement
            if not text_content:
                # Chercher dans toutes les balises de contenu
                content_elements = soup.find_all(['p', 'div', 'section', 'article'])
                for element in content_elements:
                    text = element.get_text().strip()
                    # Filtrer les textes trop courts ou qui semblent être des menus
                    if (text and len(text.split()) > 10 and 
                        not any(word in text.lower() for word in ['home', 'login', 'sign up', 'menu', 'navigation'])):
                        text_content.append(text)
            
            # 3. Si toujours pas de contenu, essayer une extraction large
            if not text_content:
                all_text = soup.get_text()
                lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                # Prendre les lignes les plus longues (probablement du contenu)
                text_content = [line for line in lines if len(line.split()) > 15][:20]
            
            if text_content:
                # Nettoyer et joindre le texte
                full_text = ' '.join(text_content)
                # Nettoyer les espaces multiples
                full_text = ' '.join(full_text.split())
                # Vérifier que le contenu est significatif
                if len(full_text.split()) > 50:
                    return full_text
                else:
                    return f"Contenu trop court extrait de {url} (seulement {len(full_text.split())} mots)"
            else:
                return f"Aucun contenu textuel significatif trouvé sur {url}"
                
        except requests.exceptions.RequestException as e:
            return f"Erreur de connexion: {str(e)}"
        except Exception as e:
            return f"Erreur lors de l'extraction: {str(e)}"

    def get_summary_model(self, language: str):
        """Retourne le modèle de résumé optimal pour la langue spécifiée"""
        lang_code = self.language_codes.get(language.lower(), "en")

        # Essayer le modèle spécifique à la langue
        if lang_code in self.summary_models:
            return self.summary_models[lang_code]

        # Fallback sur multilingue
        if "multilingual" in self.summary_models:
            return self.summary_models["multilingual"]

        # Fallback sur anglais
        if "en" in self.summary_models:
            return self.summary_models["en"]

        raise ValueError("Aucun modèle de résumé disponible")

    def get_translation_model(self, source_lang: str, target_lang: str):
        """Retourne le modèle de traduction optimal pour la paire de langues"""
        src_code = self.language_codes.get(source_lang.lower(), "en")
        tgt_code = self.language_codes.get(target_lang.lower(), "en")

        # Essayer MarianMT d'abord (spécialisé par paire)
        marian_key = f"marian_{src_code}-{tgt_code}"
        if marian_key in self.translation_models:
            return self.translation_models[marian_key]

        # Essayer M2M100 (multilingue - supporte arabe)
        if "m2m100" in self.translation_models:
            return self.translation_models["m2m100"]

        # Essayer MBART-50 (multilingue - supporte arabe)
        if "mbart50" in self.translation_models:
            return self.translation_models["mbart50"]

        # Essayer d'autres paires MarianMT inversées
        for key in self.translation_models.keys():
            if key.startswith("marian_") and f"{tgt_code}-{src_code}" in key:
                logger.info(f"Utilisation du modèle inverse {key} pour {src_code}-{tgt_code}")
                return self.translation_models[key]

        raise ValueError(f"Aucun modèle de traduction disponible pour {src_code}-{tgt_code}")

    def summarize_text(self, text: str, language: str, summary_length: str = "medium") -> str:
        """Résume le texte avec le modèle optimal pour la langue"""
        if not self.models_loaded:
            raise RuntimeError("Les modèles ne sont pas chargés")

        try:
            model = self.get_summary_model(language)

            # Paramètres de longueur optimisés
            length_params = self._get_length_parameters(summary_length)

            # Pour mT5, ajouter le préfixe
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                if 'mt5' in model.model.config.name_or_path.lower():
                    text = f"summarize: {text}"

            # Résumé avec le modèle sélectionné
            result = model(
                text,
                max_length=length_params["max_length"],
                min_length=length_params["min_length"],
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']

        except Exception as e:
            logger.error(f"Erreur lors du résumé: {e}")
            return self._simple_fallback_summary(text, summary_length)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduit le texte avec le modèle optimal"""
        if not self.models_loaded:
            raise RuntimeError("Les modèles ne sont pas chargés")

        try:
            model_info = self.get_translation_model(source_lang, target_lang)
            src_code = self.language_codes.get(source_lang.lower(), "en")
            tgt_code = self.language_codes.get(target_lang.lower(), "en")

            model_type = model_info.get("type", "unknown")

            if model_type == "m2m100":
                return self._translate_m2m100(model_info, text, src_code, tgt_code)
            elif model_type == "mbart50":
                return self._translate_mbart50(model_info, text, src_code, tgt_code)
            elif model_type == "marian":
                return self._translate_marian(model_info, text)
            else:
                # Détection automatique du type
                model = model_info.get("model")
                if model is None:
                    raise ValueError("Modèle non défini")

                model_class = str(type(model)).lower()
                if "m2m100" in model_class:
                    return self._translate_m2m100(model_info, text, src_code, tgt_code)
                elif "mbart" in model_class:
                    return self._translate_mbart50(model_info, text, src_code, tgt_code)
                elif "marian" in model_class:
                    return self._translate_marian(model_info, text)
                else:
                    raise ValueError(f"Type de modèle non supporté: {type(model)}")

        except Exception as e:
            logger.error(f"Erreur lors de la traduction: {e}")
            return self._translate_with_fallback(text, target_lang)

    def _get_length_parameters(self, summary_length: str) -> Dict:
        """Retourne les paramètres de longueur optimisés"""
        params = {
            "court": {"max_length": 80, "min_length": 30},
            "moyen": {"max_length": 150, "min_length": 50},
            "long": {"max_length": 250, "min_length": 80}
        }
        return params.get(summary_length, params["moyen"])

    def _translate_m2m100(self, model_info: Dict, text: str, src_lang: str, tgt_lang: str) -> str:
        """Traduction avec M2M100 (supporte arabe)"""
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        # Configuration des langues pour M2M100
        tokenizer.src_lang = src_lang

        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if self.device.type == "cuda":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            model = model.to(self.device)

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=512,
            num_beams=4
        )

        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _translate_mbart50(self, model_info: Dict, text: str, src_lang: str, tgt_lang: str) -> str:
        """Traduction avec MBART-50 (supporte arabe)"""
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        # Configuration des codes de langue MBART
        tokenizer.src_lang = src_lang

        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if self.device.type == "cuda":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            model = model.to(self.device)

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=512,
            num_beams=4
        )

        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _translate_marian(self, model_info: Dict, text: str) -> str:
        """Traduction avec MarianMT"""
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if self.device.type == "cuda":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            model = model.to(self.device)

        generated_tokens = model.generate(
            **encoded,
            max_length=512,
            num_beams=4
        )

        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _translate_with_fallback(self, text: str, target_lang: str) -> str:
        """Traduction avec fallback API"""
        try:
            return self._translate_with_libretranslate(text, target_lang)
        except Exception as e:
            logger.warning(f"Fallback API échoué: {e}")
            return self._simple_fallback_translation(text, target_lang)

    def _translate_with_libretranslate(self, text: str, target_lang: str) -> str:
        """Utilise LibreTranslate API"""
        import requests
        import time

        lang_code = self.language_codes.get(target_lang.lower(), "en")

        endpoints = [
            "https://translate.argosopentech.com/translate",
            "https://libretranslate.de/translate"
        ]

        for endpoint in endpoints:
            try:
                response = requests.post(
                    endpoint,
                    json={
                        "q": text,
                        "source": "auto",
                        "target": lang_code,
                        "format": "text"
                    },
                    timeout=15
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["translatedText"]
                time.sleep(1)
            except:
                continue

        raise Exception("Tous les endpoints LibreTranslate ont échoué")

    def _simple_fallback_summary(self, text: str, summary_length: str) -> str:
        """Résumé de fallback simple"""
        sentences = text.split('. ')
        length_ratio = {
            "court": 0.2,
            "moyen": 0.4,
            "long": 0.6
        }

        num_sentences = max(1, int(len(sentences) * length_ratio[summary_length]))
        summary = '. '.join(sentences[:num_sentences])

        if summary and not summary.endswith('.'):
            summary += '.'

        return summary if summary else text[:200] + "..."

    def _simple_fallback_translation(self, text: str, target_lang: str) -> str:
        """Traduction de fallback simulée"""
        translations = {
            "en": f"[English] {text}",
            "fr": f"[Français] {text}",
            "es": f"[Español] {text}",
            "de": f"[Deutsch] {text}",
            "it": f"[Italiano] {text}",
            "pt": f"[Português] {text}",
            "ar": f"[العربية] {text}",  # Arabe
            "ru": f"[Русский] {text}",  # Russe
            "zh": f"[中文] {text}",     # Chinois
            "ja": f"[日本語] {text}",   # Japonais
        }

        lang_code = self.language_codes.get(target_lang.lower(), "en")
        return translations.get(lang_code, f"[{target_lang}] {text}")

    def get_available_languages(self) -> List[str]:
        """Retourne la liste des langues supportées"""
        return list(set([lang for lang in self.language_codes.keys() if len(lang) > 2]))

    def get_model_status(self) -> Dict:
        """Retourne le statut de tous les modèles"""
        status = {
            "models_loaded": self.models_loaded,
            "device": str(self.device),
            "summary_models": list(self.summary_models.keys()) if self.models_loaded else [],
            "translation_models": [k for k in self.translation_models.keys() if not k.startswith('marian_')] if self.models_loaded else [],
            "marian_models": [k for k in self.translation_models.keys() if k.startswith('marian_')] if self.models_loaded else [],
            "supported_languages": self.get_available_languages()
        }
        return status

# Instance globale
def get_multilingual_models():
    return ExtendedMultilingualModels()
