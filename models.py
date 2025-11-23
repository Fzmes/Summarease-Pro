import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer
)
import gc

logger = logging.getLogger(__name__)


class MultilingualModels:
    """
    Version optimisée pour SummarEase Pro avec lazy-loading et gestion mémoire.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.loaded = False

        # Containers pour modèles et tokenizers
        self.models = {}
        self.tokenizers = {}

        logger.info("🧠 MultilingualModels initialisé (Lazy-loading activé)")

    def _load_specific_model(self, model_name):
        """Charge un modèle spécifique de manière optimisée"""
        try:
            if model_name == "barthez":
                self.tokenizers["barthez"] = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
                self.models["barthez"] = AutoModelForSeq2SeqLM.from_pretrained(
                    "moussaKam/barthez-orangesum-abstract"
                ).to(self.device)

            elif model_name == "bart_en":
                self.tokenizers["bart_en"] = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
                self.models["bart_en"] = AutoModelForSeq2SeqLM.from_pretrained(
                    "facebook/bart-large-cnn"
                ).to(self.device)

            elif model_name == "mt5":
                self.tokenizers["mt5"] = AutoTokenizer.from_pretrained("google/mt5-small")
                self.models["mt5"] = AutoModelForSeq2SeqLM.from_pretrained(
                    "google/mt5-small"
                ).to(self.device)

            elif model_name == "mbart":
                self.tokenizers["mbart"] = MBart50TokenizerFast.from_pretrained(
                    "facebook/mbart-large-50-many-to-many-mmt"
                )
                self.models["mbart"] = MBartForConditionalGeneration.from_pretrained(
                    "facebook/mbart-large-50-many-to-many-mmt"
                ).to(self.device)

            elif model_name == "m2m":
                self.tokenizers["m2m"] = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
                self.models["m2m"] = AutoModelForSeq2SeqLM.from_pretrained(
                    "facebook/m2m100_418M"
                ).to(self.device)

            elif model_name == "marian_fr_en":
                self.tokenizers["marian_fr_en"] = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-fr-en"
                )
                self.models["marian_fr_en"] = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-fr-en"
                ).to(self.device)

            elif model_name == "marian_en_fr":
                self.tokenizers["marian_en_fr"] = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-fr"
                )
                self.models["marian_en_fr"] = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-fr"
                ).to(self.device)

            logger.info(f"✅ Modèle {model_name} chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle {model_name}: {str(e)}")
            return False

    def load_essential_models(self):
        """Charge uniquement les modèles essentiels pour réduire l'empreinte mémoire"""
        if self.loaded:
            return True
            
        logger.info("🚀 Chargement des modèles essentiels...")
        
        essential_models = ["barthez", "bart_en", "m2m"]
        success_count = 0
        
        for model_name in essential_models:
            if self._load_specific_model(model_name):
                success_count += 1
        
        self.loaded = success_count > 0
        logger.info(f"✅ {success_count}/{len(essential_models)} modèles essentiels chargés")
        return self.loaded

    def load_all(self):
        """Charge tous les modèles (méthode originale préservée)"""
        if self.loaded:
            return True
            
        logger.info("🚀 Chargement de tous les modèles...")
        
        all_models = [
            "barthez", "bart_en", "mt5", "mbart", 
            "m2m", "marian_fr_en", "marian_en_fr"
        ]
        success_count = 0
        
        for model_name in all_models:
            if self._load_specific_model(model_name):
                success_count += 1
        
        self.loaded = success_count > 0
        logger.info(f"✅ {success_count}/{len(all_models)} modèles chargés")
        return self.loaded

    def get_model_status(self):
        return {
            "models_loaded": self.loaded,
            "device": self.device,
            "loaded_models_count": len(self.models)
        }

    def summarize_text(self, text, lang="français", length="moyen"):
        """Résume le texte avec gestion d'erreurs améliorée"""
        if not self.loaded:
            if not self.load_essential_models():
                raise Exception("Impossible de charger les modèles essentiels")

        try:
            # Validation de la longueur du texte - CORRIGÉ : caractères au lieu de mots
            if len(text.strip()) < 50:
                raise ValueError("Le texte est trop court pour être résumé (minimum 50 caractères)")
            
            # Sélection du modèle adapté
            if lang == "français":
                model_key = "barthez"
            elif lang == "anglais":
                model_key = "bart_en"
            else:
                model_key = "mt5"  # Modèle multilingue par défaut

            # Chargement à la demande si nécessaire
            if model_key not in self.models:
                if not self._load_specific_model(model_key):
                    # Fallback vers un modèle disponible
                    if "barthez" in self.models:
                        model_key = "barthez"
                    elif "mt5" in self.models:
                        model_key = "mt5"
                    else:
                        raise Exception("Aucun modèle de résumé disponible")

            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]

            # Configuration de la longueur du résumé
            max_len = 120 if length == "court" else 220 if length == "moyen" else 350

            # Tokenization avec gestion des textes longs
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Génération du résumé - SUPPRIMER early_stopping qui cause des warnings
            output = model.generate(
                **inputs, 
                max_length=max_len,
                num_beams=4
                # early_stopping=True  # Supprimé car cause des warnings
            )

            return tokenizer.decode(output[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Erreur lors du résumé: {str(e)}")
            raise

    def translate_text(self, text, src_lang, tgt_lang):
        """Traduit le texte avec gestion d'erreurs améliorée"""
        if not self.loaded:
            if not self.load_essential_models():
                raise Exception("Impossible de charger les modèles essentiels")

        try:
            # Validation
            if src_lang == tgt_lang:
                return text  # Pas de traduction nécessaire

            # Sélection du modèle de traduction
            if src_lang == "français" and tgt_lang == "anglais":
                model_key = "marian_fr_en"
            elif src_lang == "anglais" and tgt_lang == "français":
                model_key = "marian_en_fr"
            else:
                model_key = "m2m"  # Modèle multilingue

            # Chargement à la demande si nécessaire
            if model_key not in self.models:
                if not self._load_specific_model(model_key):
                    raise Exception(f"Modèle de traduction {src_lang}->{tgt_lang} non disponible")

            tok = self.tokenizers[model_key]
            mod = self.models[model_key]

            # Configuration spécifique pour m2m
            if model_key == "m2m":
                lang_map = {
                    "français": "fr", "anglais": "en", "espagnol": "es", 
                    "allemand": "de", "arabe": "ar"
                }
                tok.src_lang = lang_map.get(src_lang, "fr")

            # Tokenization et traduction
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            out = mod.generate(**inputs)

            return tok.decode(out[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Erreur lors de la traduction {src_lang}->{tgt_lang}: {str(e)}")
            raise

    def cleanup_memory(self):
        """Nettoie la mémoire GPU et libère les ressources"""
        try:
            # Libération des modèles
            for model in self.models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
            
            # Nettoyage GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Nettoyage mémoire Python
            gc.collect()
            
            logger.info("🧹 Mémoire nettoyée avec succès")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de nettoyer complètement la mémoire: {str(e)}")


def get_multilingual_models(device="cpu"):
    return MultilingualModels(device=device)