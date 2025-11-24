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
from typing import Dict, List
import warnings
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedMultilingualModels:
    """Gestionnaire de modèles multilingues étendu avec support arabe, résumé & traduction."""

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models_loaded = False

        # Codes langues
        self.language_codes = {
            "français": "fr", "fr": "fr",
            "anglais": "en", "en": "en",
            "espagnol": "es", "es": "es",
            "allemand": "de", "de": "de",
            "italien": "it", "it": "it",
            "portugais": "pt", "pt": "pt",
            "néerlandais": "nl", "nl": "nl",
            "russe": "ru", "ru": "ru",
            "chinois": "zh", "zh": "zh",
            "japonais": "ja", "ja": "ja",
            "arabe": "ar", "ar": "ar",
            "coréen": "ko", "ko": "ko",
            "hindi": "hi"
        }

        # Load all models
        self._load_models()

    def _setup_device(self, device: str):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    # -----------------------------------------------------
    # LOADING MODELS
    # -----------------------------------------------------

    def _load_models(self):
        try:
            logger.info("🚀 Chargement des modèles...")
            self.summary_models = self._load_summary_models()
            self.translation_models = self._load_translation_models()
            self.models_loaded = True
            logger.info("✅ Modèles chargés avec succès!")
        except Exception as e:
            logger.error(f"❌ Erreur de chargement : {e}")
            self.models_loaded = False

    def _load_summary_models(self) -> Dict:
        models = {}

        # FR
        try:
            models["fr"] = pipeline(
                "summarization",
                model="moussaKam/barthez-orangesum-abstract",
                tokenizer="moussaKam/barthez-orangesum-abstract",
                device=0 if self.device.type == "cuda" else -1
            )
        except Exception:
            logger.warning("⚠️ Modèle résumé FR indisponible")

        # EN
        try:
            models["en"] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn",
                device=0 if self.device.type == "cuda" else -1
            )
        except Exception:
            logger.warning("⚠️ Modèle résumé EN indisponible")

        # Multilingual fallback
        try:
            models["multilingual"] = pipeline(
                "summarization",
                model="google/mt5-small",
                tokenizer="google/mt5-small",
                device=0 if self.device.type == "cuda" else -1
            )
        except Exception:
            logger.warning("⚠️ Modèle résumé multilingue indisponible")

        return models

    def _load_translation_models(self) -> Dict:
        models = {}

        # M2M100
        try:
            models["m2m100"] = {
                "model": AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M"),
                "tokenizer": AutoTokenizer.from_pretrained("facebook/m2m100_418M"),
                "type": "m2m100"
            }
        except:
            pass

        # MBART50
        try:
            models["mbart50"] = {
                "model": MBartForConditionalGeneration.from_pretrained(
                    "facebook/mbart-large-50-many-to-many-mmt"
                ),
                "tokenizer": MBart50TokenizerFast.from_pretrained(
                    "facebook/mbart-large-50-many-to-many-mmt"
                ),
                "type": "mbart50"
            }
        except:
            pass

        # MarianMT models
        pairs = {
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "en-de": "Helsinki-NLP/opus-mt-en-de",
            "de-en": "Helsinki-NLP/opus-mt-de-en",
            "en-ar": "Helsinki-NLP/opus-mt-en-ar",
            "ar-en": "Helsinki-NLP/opus-mt-ar-en"
        }

        for pair, model_name in pairs.items():
            try:
                models[f"marian_{pair}"] = {
                    "model": MarianMTModel.from_pretrained(model_name),
                    "tokenizer": MarianTokenizer.from_pretrained(model_name),
                    "type": "marian"
                }
            except:
                pass

        return models

    # -----------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------

    def load_essential_models(self):
        """Used by app.py – always true since everything loads in __init__."""
        return self.models_loaded

    def cleanup_memory(self):
        """Clean GPU/CPU memory."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------
    # MODEL ACCESS
    # -----------------------------------------------------

    def get_summary_model(self, language: str):
        code = self.language_codes.get(language.lower(), "en")
        if code in self.summary_models:
            return self.summary_models[code]
        if "multilingual" in self.summary_models:
            return self.summary_models["multilingual"]
        return self.summary_models["en"]

    def get_translation_model(self, src: str, tgt: str):
        s = self.language_codes.get(src.lower(), "en")
        t = self.language_codes.get(tgt.lower(), "en")

        key = f"marian_{s}-{t}"
        if key in self.translation_models:
            return self.translation_models[key]

        if "m2m100" in self.translation_models:
            return self.translation_models["m2m100"]

        if "mbart50" in self.translation_models:
            return self.translation_models["mbart50"]

        raise ValueError(f"No translation model for {s}->{t}")

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    def summarize_text(self, text, language, summary_length="moyen"):
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")

        params = {
            "court": (30, 80),
            "moyen": (50, 150),
            "long": (80, 250)
        }.get(summary_length, (50, 150))

        min_l, max_l = params
        model = self.get_summary_model(language)

        try:
            result = model(
                text,
                max_length=max_l,
                min_length=min_l,
                do_sample=False,
                truncation=True
            )
            return result[0]['summary_text']
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return text[:400] + "..."

    # -----------------------------------------------------
    # TRANSLATION
    # -----------------------------------------------------

    def translate_text(self, text, src, tgt):
        model_info = self.get_translation_model(src, tgt)
        typ = model_info["type"]

        if typ == "marian":
            return self._translate_marian(model_info, text)

        if typ == "m2m100":
            return self._translate_m2m100(model_info, text, src, tgt)

        if typ == "mbart50":
            return self._translate_mbart50(model_info, text, src, tgt)

        raise ValueError("Unsupported model type")

    def _translate_marian(self, info, text):
        t = info["tokenizer"]
        m = info["model"]
        enc = t(text, return_tensors="pt", truncation=True)
        gen = m.generate(**enc, max_length=512, num_beams=4)
        return t.decode(gen[0], skip_special_tokens=True)

    def _translate_m2m100(self, info, text, src, tgt):
        tokenizer = info["tokenizer"]
        model = info["model"]

        src_code = self.language_codes[src.lower()]
        tgt_code = self.language_codes[tgt.lower()]

        tokenizer.src_lang = src_code
        encoded = tokenizer(text, return_tensors="pt", truncation=True)

        gen = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
            max_length=512
        )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    def _translate_mbart50(self, info, text, src, tgt):
        tokenizer = info["tokenizer"]
        model = info["model"]

        src_code = self.language_codes[src.lower()]
        tgt_code = self.language_codes[tgt.lower()]

        tokenizer.src_lang = src_code
        enc = tokenizer(text, return_tensors="pt", truncation=True)

        gen = model.generate(
            **enc,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
            max_length=512
        )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    # -----------------------------------------------------
    # SYSTEM INFO
    # -----------------------------------------------------

    def get_available_languages(self):
        return list({lang for lang in self.language_codes.keys() if len(lang) > 2})

    def get_model_status(self):
        return {
            "models_loaded": self.models_loaded,
            "device": str(self.device),
            "summary_models": list(self.summary_models.keys()),
            "translation_models": list(self.translation_models.keys()),
            "loaded_models_count": len(self.summary_models) + len(self.translation_models),
            "supported_languages": self.get_available_languages()
        }


# ---------------------------------------------------------
# GLOBAL INSTANCE + ENTRYPOINT FOR app.py
# ---------------------------------------------------------

_multilingual_instance = ExtendedMultilingualModels()

def get_multilingual_models():
    return _multilingual_instance
