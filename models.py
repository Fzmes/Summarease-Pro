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

logger = logging.getLogger(__name__)


class MultilingualModels:
    """
    Version optimisée pour SummarEase Pro.
    - Aucun modèle n'est chargé dans __init__
    - Lazy-loading automatique
    - Compatible avec summarize_text() + translate_text()
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.loaded = False

        # Containers
        self.models = {}
        self.tokenizers = {}

        logger.info("🧠 MultilingualModels initialisé (Lazy-loading activé)")

    # ======================================================
    # LAZY LOAD PRINCIPAL — appelle TOUS les téléchargements
    # ======================================================
    def load_all(self):
        if self.loaded:
            return

        logger.info("🚀 Chargement des modèles multilingues…")

        # ==========================
        # BARTHEZ (Résumé FR)
        # ==========================
        self.tokenizers["barthez"] = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
        self.models["barthez"] = AutoModelForSeq2SeqLM.from_pretrained(
            "moussaKam/barthez-orangesum-abstract"
        ).to(self.device)

        # ==========================
        # BART-large-CNN (Résumé EN)
        # ==========================
        self.tokenizers["bart_en"] = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.models["bart_en"] = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn"
        ).to(self.device)

        # ==========================
        # mT5-small (Résumé Multi)
        # ==========================
        self.tokenizers["mt5"] = AutoTokenizer.from_pretrained("google/mt5-small")
        self.models["mt5"] = AutoModelForSeq2SeqLM.from_pretrained(
            "google/mt5-small"
        ).to(self.device)

        # ==========================
        # MBART-50 (Résumé + Traduction Multi)
        # ==========================
        self.tokenizers["mbart"] = MBart50TokenizerFast.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        )
        self.models["mbart"] = MBartForConditionalGeneration.from_pretrained(
            "facebook/mbart-large-50-many-to-many-mmt"
        ).to(self.device)

        # ==========================
        # M2M100 (Traduction Multi)
        # ==========================
        self.tokenizers["m2m"] = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        self.models["m2m"] = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/m2m100_418M"
        ).to(self.device)

        # ==========================
        # MarianMT (FR<->EN)
        # ==========================
        self.tokenizers["marian_fr_en"] = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-fr-en"
        )
        self.models["marian_fr_en"] = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-fr-en"
        ).to(self.device)

        self.tokenizers["marian_en_fr"] = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        )
        self.models["marian_en_fr"] = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        ).to(self.device)

        self.loaded = True
        logger.info("✅ Tous les modèles SummarEase chargés avec succès.")

    # ======================================================
    # API : Fonctions nécessaires à ton app.py
    # ======================================================

    def get_model_status(self):
        return {
            "models_loaded": self.loaded,
            "device": self.device,
        }

    # ============
    # Résumé
    # ============
    def summarize_text(self, text, lang="fr", length="moyen"):
        if not self.loaded:
            self.load_all()

        # Sélection du modèle
        if lang == "français":
            model = self.models["barthez"]
            tokenizer = self.tokenizers["barthez"]
        elif lang == "anglais":
            model = self.models["bart_en"]
            tokenizer = self.tokenizers["bart_en"]
        else:
            model = self.models["mt5"]
            tokenizer = self.tokenizers["mt5"]

        max_len = 120 if length == "court" else 220 if length == "moyen" else 350

        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        output = model.generate(**inputs, max_length=max_len)

        return tokenizer.decode(output[0], skip_special_tokens=True)

    # ============
    # Traduction
    # ============
    def translate_text(self, text, src_lang, tgt_lang):
        if not self.loaded:
            self.load_all()

        # FR -> EN
        if src_lang == "français" and tgt_lang == "anglais":
            tok = self.tokenizers["marian_fr_en"]
            mod = self.models["marian_fr_en"]

        # EN -> FR
        elif src_lang == "anglais" and tgt_lang == "français":
            tok = self.tokenizers["marian_en_fr"]
            mod = self.models["marian_en_fr"]

        else:
            # fallback multilingue
            tok = self.tokenizers["m2m"]
            mod = self.models["m2m"]
            tok.src_lang = src_lang[:2]  # ex: "français" -> "fr"

        inputs = tok(text, return_tensors="pt", truncation=True).to(self.device)
        out = mod.generate(**inputs)

        return tok.decode(out[0], skip_special_tokens=True)


# ===========================================================
# Factory function (tu l’utilises dans app.py) — inchangée
# ===========================================================
def get_multilingual_models(device="cpu"):
    return MultilingualModels(device=device)
