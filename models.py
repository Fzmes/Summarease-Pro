import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
    LEDTokenizer,
    LEDForConditionalGeneration
)
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import requests
from bs4 import BeautifulSoup
import re
import gc
from collections import Counter

warnings.filterwarnings("ignore")

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScientificArticleProcessor:
    """Processeur spécialisé pour articles scientifiques et documents longs (1-90 pages)"""

    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models_loaded = False

        # Mapping étendu des langues scientifiques
        self.language_codes = {
            "français": "fr", "fr": "fr", "french": "fr",
            "anglais": "en", "en": "en", "english": "en",
            "espagnol": "es", "es": "es", "spanish": "es",
            "allemand": "de", "de": "de", "german": "de",
            "italien": "it", "it": "it", "italian": "it",
            "portugais": "pt", "pt": "pt", "portuguese": "pt",
            "arabe": "ar", "ar": "ar", "arabic": "ar",
            "russe": "ru", "ru": "ru", "russian": "ru",
            "chinois": "zh", "zh": "zh", "chinese": "zh",
            "japonais": "ja", "ja": "ja", "japanese": "ja"
        }

        # Configuration des modèles spécialisés pour documents longs
        self.model_configs = {
            # Modèles de résumé pour contexte long
            "scientific_led": {
                "name": "allenai/led-base-16384",
                "type": "scientific_summary",
                "max_tokens": 16384,
                "priority": 1
            },
            "scientific_bart": {
                "name": "facebook/bart-large-cnn",
                "type": "summary",
                "max_tokens": 1024,
                "priority": 2
            },
            "scientific_mt5": {
                "name": "google/mt5-small",
                "type": "multilingual_summary",
                "max_tokens": 512,
                "priority": 3
            },
            # Modèles de traduction scientifique
            "translate_m2m": {
                "name": "facebook/m2m100_418M",
                "type": "translation",
                "max_tokens": 512,
                "priority": 1
            },
            "translate_mbart": {
                "name": "facebook/mbart-large-50-many-to-many-mmt",
                "type": "translation",
                "max_tokens": 512,
                "priority": 2
            }
        }

        self.loaded_models = {}
        self.loaded_tokenizers = {}
        
        self._load_scientific_models()

    def _setup_device(self, device: str) -> torch.device:
        """Configure le device avec optimisation pour longs documents"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _load_scientific_models(self):
        """Charge les modèles spécialisés pour articles scientifiques"""
        try:
            logger.info("🔬 Chargement des modèles scientifiques pour documents longs...")

            # Modèle LED pour contexte long (16K tokens)
            try:
                self.loaded_tokenizers["scientific_led"] = LEDTokenizer.from_pretrained("allenai/led-base-16384")
                self.loaded_models["scientific_led"] = LEDForConditionalGeneration.from_pretrained(
                    "allenai/led-base-16384",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device)
                logger.info("✅ Modèle LED (16K tokens) chargé pour documents longs")
            except Exception as e:
                logger.warning(f"Modèle LED non disponible: {e}")

            # Modèles de résumé standard
            try:
                self.loaded_models["scientific_bart"] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    tokenizer="facebook/bart-large-cnn",
                    device=self.device,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                )
                logger.info("✅ Modèle BART scientifique chargé")
            except Exception as e:
                logger.warning(f"Modèle BART non disponible: {e}")

            # Modèle multilingue
            try:
                self.loaded_models["scientific_mt5"] = pipeline(
                    "summarization",
                    model="google/mt5-small",
                    tokenizer="google/mt5-small",
                    device=self.device
                )
                logger.info("✅ Modèle mT5 multilingue chargé")
            except Exception as e:
                logger.warning(f"Modèle mT5 non disponible: {e}")

            # Modèles de traduction
            try:
                self.loaded_tokenizers["translate_m2m"] = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
                self.loaded_models["translate_m2m"] = AutoModelForSeq2SeqLM.from_pretrained(
                    "facebook/m2m100_418M",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                ).to(self.device)
                logger.info("✅ Modèle M2M100 de traduction chargé")
            except Exception as e:
                logger.warning(f"Modèle M2M100 non disponible: {e}")

            self.models_loaded = len(self.loaded_models) > 0
            logger.info("✅ Modèles scientifiques chargés avec succès!")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles scientifiques: {e}")
            self.models_loaded = False

    def preprocess_scientific_text(self, text: str) -> Dict:
        """Prétraitement avancé pour articles scientifiques"""
        # Nettoyage du texte
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = re.sub(r'\[\d+\]', '', text)  # Références [1], [2], etc.
        text = re.sub(r'\([^)]*\)', '', text)  # Parenthèses
        text = re.sub(r'Figure\s+\d+[:\-]\s*', '', text, flags=re.IGNORECASE)  # Références aux figures
        
        # Détection des sections scientifiques
        sections = self._extract_scientific_sections(text)
        
        # Métriques
        word_count = len(text.split())
        char_count = len(text)
        estimated_pages = max(1, word_count // 500)  # ~500 mots par page
        
        return {
            "cleaned_text": text.strip(),
            "sections": sections,
            "metrics": {
                "word_count": word_count,
                "char_count": char_count,
                "estimated_pages": estimated_pages,
                "is_long_document": word_count > 3000
            },
            "language": self._detect_scientific_language(text)
        }

    def _extract_scientific_sections(self, text: str) -> Dict:
        """Extrait les sections typiques d'un article scientifique"""
        sections = {
            "abstract": "",
            "introduction": "",
            "methodology": "",
            "results": "",
            "discussion": "",
            "conclusion": "",
            "references": ""
        }
        
        # Patterns pour sections scientifiques
        patterns = {
            "abstract": r'(abstract|summary|résumé)[\s:\-]*\n*(.*?)(?=\n\s*\n|\n\s*1\.|\n\s*introduction|$)',
            "introduction": r'(1\.)?\s*introduction[\s:\-]*\n*(.*?)(?=\n\s*\n|\n\s*2\.|\n\s*method|$)',
            "methodology": r'(2\.)?\s*(method|methodology|materials and methods)[\s:\-]*\n*(.*?)(?=\n\s*\n|\n\s*3\.|\n\s*results|$)',
            "results": r'(3\.)?\s*(results|findings)[\s:\-]*\n*(.*?)(?=\n\s*\n|\n\s*4\.|\n\s*discussion|$)',
            "discussion": r'(4\.)?\s*discussion[\s:\-]*\n*(.*?)(?=\n\s*\n|\n\s*5\.|\n\s*conclusion|$)',
            "conclusion": r'(5\.)?\s*conclusion[\s:\-]*\n*(.*?)(?=\n\s*\n|\n\s*references|$)',
            "references": r'(references|bibliography)[\s:\-]*\n*(.*?)(?=$)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(2 if section == "abstract" else 3 if section in ["methodology", "results"] else 2).strip()
                
        return sections

    def _detect_scientific_language(self, text: str) -> str:
        """Détection de la langue basée sur le vocabulaire scientifique"""
        sample = text[:2000].lower()
        
        language_keywords = {
            "en": ["the", "this", "study", "research", "method", "results", "conclusion", "analysis"],
            "fr": ["étude", "recherche", "méthode", "résultats", "conclusion", "analyse", "cette"],
            "es": ["estudio", "investigación", "método", "resultados", "conclusión", "análisis", "este"],
            "de": ["studie", "forschung", "methode", "ergebnisse", "schlussfolgerung", "analyse", "diese"]
        }
        
        scores = {}
        for lang, keywords in language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sample)
            scores[lang] = score
            
        detected = max(scores.items(), key=lambda x: x[1])
        lang_map = {"en": "anglais", "fr": "français", "es": "espagnol", "de": "allemand"}
        return lang_map.get(detected[0], "anglais")

    def chunk_scientific_text(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Découpe le texte scientifique en chunks intelligents avec chevauchement"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Sauvegarder le chunk actuel
                chunks.append(' '.join(current_chunk))
                
                # Créer un nouveau chunk avec chevauchement
                overlap_sentences = current_chunk[-max(1, len(current_chunk) // 4):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in overlap_sentences) + sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        logger.info(f"📄 Texte découpé en {len(chunks)} chunks")
        return chunks

    def summarize_long_article(self, text: str, language: str = "anglais", 
                             summary_type: str = "structured") -> Dict:
        """
        Résumé spécialisé pour articles scientifiques longs
        Types: structured, abstract, key_points, comprehensive
        """
        if not self.models_loaded:
            raise RuntimeError("Les modèles ne sont pas chargés")

        try:
            # Prétraitement et analyse
            processed = self.preprocess_scientific_text(text)
            logger.info(f"📊 Document analysé: {processed['metrics']}")

            # Stratégie basée sur la longueur
            if processed["metrics"]["word_count"] > 10000:
                return self._summarize_very_long_article(processed, language, summary_type)
            elif processed["metrics"]["word_count"] > 3000:
                return self._summarize_long_article(processed, language, summary_type)
            else:
                return self._summarize_medium_article(processed, language, summary_type)

        except Exception as e:
            logger.error(f"❌ Erreur résumé scientifique: {e}")
            return self._fallback_scientific_summary(text, language)

    def _summarize_very_long_article(self, processed: Dict, language: str, summary_type: str) -> Dict:
        """Résumé d'articles très longs (>10,000 mots)"""
        text = processed["cleaned_text"]
        chunks = self.chunk_scientific_text(text, chunk_size=3000, overlap=300)
        
        chunk_summaries = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            logger.info(f"📝 Traitement chunk {i+1}/{total_chunks}")
            
            try:
                if "scientific_led" in self.loaded_models:
                    summary = self._summarize_with_led(chunk)
                else:
                    summary = self._summarize_with_bart(chunk)
                    
                chunk_summaries.append(summary)
                
                # Libération mémoire périodique
                if i % 2 == 0:
                    self.cleanup_memory()
                    
            except Exception as e:
                logger.warning(f"Chunk {i+1} échoué: {e}")
                continue

        # Combinaison hiérarchique
        if len(chunk_summaries) > 1:
            combined_text = ' '.join(chunk_summaries)
            final_summary = self._create_structured_summary(combined_text, summary_type)
        else:
            final_summary = chunk_summaries[0] if chunk_summaries else "Résumé non disponible"

        return {
            "summary": final_summary,
            "sections_analyzed": [k for k, v in processed["sections"].items() if v],
            "original_metrics": processed["metrics"],
            "summary_type": summary_type,
            "chunks_processed": len(chunks),
            "processing_strategy": "very_long_hierarchical"
        }

    def _summarize_long_article(self, processed: Dict, language: str, summary_type: str) -> Dict:
        """Résumé d'articles longs (3,000-10,000 mots)"""
        text = processed["cleaned_text"]
        
        # Utiliser LED si disponible pour le contexte long
        if "scientific_led" in self.loaded_models:
            summary = self._summarize_with_led(text)
        else:
            # Fallback: résumé par sections
            summary = self._summarize_by_sections(processed)
        
        structured_summary = self._create_structured_summary(summary, summary_type)

        return {
            "summary": structured_summary,
            "sections_analyzed": [k for k, v in processed["sections"].items() if v],
            "original_metrics": processed["metrics"],
            "summary_type": summary_type,
            "processing_strategy": "long_direct"
        }

    def _summarize_medium_article(self, processed: Dict, language: str, summary_type: str) -> Dict:
        """Résumé d'articles de longueur moyenne"""
        text = processed["cleaned_text"]
        
        if language.lower() in ["anglais", "english"] and "scientific_bart" in self.loaded_models:
            summary = self._summarize_with_bart(text)
        elif "scientific_mt5" in self.loaded_models:
            summary = self._summarize_with_mt5(text, language)
        else:
            summary = self._summarize_with_bart(text)
        
        structured_summary = self._create_structured_summary(summary, summary_type)

        return {
            "summary": structured_summary,
            "sections_analyzed": [k for k, v in processed["sections"].items() if v],
            "original_metrics": processed["metrics"],
            "summary_type": summary_type,
            "processing_strategy": "medium_direct"
        }

    def _summarize_with_led(self, text: str) -> str:
        """Utilise LED pour les longs contextes (16K tokens)"""
        tokenizer = self.loaded_tokenizers["scientific_led"]
        model = self.loaded_models["scientific_led"]
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # LED peut gérer jusqu'à 16384
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                min_length=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _summarize_with_bart(self, text: str) -> str:
        """Utilise BART pour résumé scientifique"""
        model = self.loaded_models["scientific_bart"]
        
        result = model(
            text,
            max_length=300,
            min_length=100,
            do_sample=False
        )
        return result[0]['summary_text']

    def _summarize_with_mt5(self, text: str, language: str) -> str:
        """Utilise mT5 pour résumé multilingue"""
        model = self.loaded_models["scientific_mt5"]
        
        # Ajouter le préfixe de langue si nécessaire
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            if 'mt5' in model.model.config.name_or_path.lower():
                text = f"summarize: {text}"

        result = model(
            text,
            max_length=300,
            min_length=100,
            do_sample=False
        )
        return result[0]['summary_text']

    def _summarize_by_sections(self, processed: Dict) -> str:
        """Résumé basé sur l'extraction des sections"""
        sections = processed["sections"]
        summary_parts = []
        
        # Priorité des sections pour le résumé
        priority_sections = ["abstract", "conclusion", "results", "introduction"]
        
        for section in priority_sections:
            if sections[section] and len(sections[section].split()) > 10:
                summary_parts.append(f"{section.upper()}: {sections[section]}")
        
        if summary_parts:
            return "\n\n".join(summary_parts[:3])  # Limiter à 3 sections
        else:
            # Fallback: premières et dernières phrases
            sentences = processed["cleaned_text"].split('. ')
            if len(sentences) > 6:
                return '. '.join(sentences[:3] + sentences[-3:]) + '.'
            else:
                return processed["cleaned_text"]

    def _create_structured_summary(self, summary: str, summary_type: str) -> str:
        """Crée un résumé structuré selon le type demandé"""
        if summary_type == "structured":
            return f"📊 RÉSUMÉ STRUCTURÉ\n\n{summary}"
        elif summary_type == "abstract":
            return f"📋 ABSTRACT\n\n{summary}"
        elif summary_type == "key_points":
            # Extraction des points clés
            sentences = summary.split('. ')
            key_points = [s.strip() for s in sentences if len(s.split()) > 5]
            return "🎯 POINTS CLÉS\n\n• " + "\n• ".join(key_points[:7])
        elif summary_type == "comprehensive":
            return f"🔍 ANALYSE COMPLÈTE\n\n{summary}"
        else:
            return summary

    def translate_scientific_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduction spécialisée pour texte scientifique"""
        if not self.models_loaded:
            raise RuntimeError("Les modèles ne sont pas chargés")

        try:
            # Pour les textes très longs, découpage
            if len(text.split()) > 2000:
                return self._translate_long_scientific_text(text, source_lang, target_lang)
            else:
                return self._translate_short_scientific_text(text, source_lang, target_lang)

        except Exception as e:
            logger.error(f"❌ Erreur traduction scientifique: {e}")
            return self._fallback_translation(text, target_lang)

    def _translate_long_scientific_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduction de textes scientifiques très longs"""
        chunks = self.chunk_scientific_text(text, chunk_size=1500, overlap=100)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"🌍 Traduction chunk {i+1}/{len(chunks)}")
            try:
                translated_chunk = self._translate_short_scientific_text(chunk, source_lang, target_lang)
                translated_chunks.append(translated_chunk)
                
                # Libération mémoire
                if i % 3 == 0:
                    self.cleanup_memory()
                    
            except Exception as e:
                logger.warning(f"Traduction chunk {i+1} échouée: {e}")
                translated_chunks.append(f"[Erreur de traduction: {str(e)}]")
        
        return ' '.join(translated_chunks)

    def _translate_short_scientific_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Traduction de textes scientifiques courts"""
        if "translate_m2m" not in self.loaded_models:
            raise ValueError("Modèle de traduction non disponible")

        model = self.loaded_models["translate_m2m"]
        tokenizer = self.loaded_tokenizers["translate_m2m"]
        
        src_code = self.language_codes.get(source_lang.lower(), "en")
        tgt_code = self.language_codes.get(target_lang.lower(), "en")
        
        tokenizer.src_lang = src_code
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_code),
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def extract_scientific_metadata(self, text: str) -> Dict:
        """Extrait les métadonnées d'un article scientifique"""
        processed = self.preprocess_scientific_text(text)
        
        # Détection des domaines scientifiques
        domains = self._detect_scientific_domains(text)
        
        # Mots clés (simplifié)
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        word_freq = Counter(words)
        keywords = [word for word, count in word_freq.most_common(15) if count > 2 and len(word) > 4]
        
        return {
            "domains": domains,
            "keywords": keywords[:10],
            "sections_present": [section for section, content in processed["sections"].items() if content],
            "language": processed["language"],
            **processed["metrics"]
        }

    def _detect_scientific_domains(self, text: str) -> List[str]:
        """Détection des domaines scientifiques basée sur le vocabulaire"""
        text_lower = text.lower()
        domains = []
        
        domain_keywords = {
            "Biologie/Médecine": ["cell", "dna", "protein", "gene", "clinical", "patient", "medical", "health", "disease"],
            "Informatique/IA": ["algorithm", "computer", "software", "data", "network", "learning", "neural", "model", "system"],
            "Physique": ["quantum", "particle", "energy", "physics", "wave", "force", "atomic", "nuclear"],
            "Chimie": ["chemical", "molecule", "reaction", "compound", "atomic", "organic", "synthesis"],
            "Mathématiques": ["equation", "theorem", "function", "mathematical", "calculation", "formula", "proof"],
            "Ingénierie": ["engineering", "design", "system", "structure", "material", "mechanical", "electrical"],
            "Sciences Sociales": ["social", "behavior", "psychological", "society", "human", "cultural", "economic"]
        }
        
        for domain, keywords in domain_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:  # Au moins 2 mots clés du domaine
                domains.append(domain)
                
        return domains if domains else ["Sciences Générales"]

    def scrape_web_content(self, url: str) -> str:
        """Scraping web optimisé pour articles scientifiques"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Nettoyage spécifique aux articles scientifiques
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "menu", "button"]):
                element.decompose()
            
            # Stratégies d'extraction pour contenu scientifique
            content_selectors = [
                'article', '.article-content', '.research-paper', '.scientific-content',
                '.paper-body', '.main-content', '[role="main"]', '.content'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            if not content:
                content = soup.find('body')
            
            text = content.get_text(strip=True, separator=' ')
            text = re.sub(r'\s+', ' ', text)
            
            # Vérification de la qualité du contenu
            if len(text.split()) < 100:
                return f"Contenu insuffisant extrait de {url} (seulement {len(text.split())} mots)"
            
            return text
            
        except Exception as e:
            return f"Erreur scraping: {str(e)}"

    def _fallback_scientific_summary(self, text: str, language: str) -> Dict:
        """Résumé de fallback pour articles scientifiques"""
        processed = self.preprocess_scientific_text(text)
        summary = self._summarize_by_sections(processed)
        
        return {
            "summary": summary,
            "sections_analyzed": ["fallback"],
            "original_metrics": processed["metrics"],
            "summary_type": "fallback",
            "processing_strategy": "fallback_extraction"
        }

    def _fallback_translation(self, text: str, target_lang: str) -> str:
        """Traduction de fallback"""
        return f"[{target_lang}] {text}"

    def cleanup_memory(self):
        """Nettoyage mémoire agressif pour documents longs"""
        try:
            # Libération des modèles de la mémoire GPU
            for model in self.loaded_models.values():
                if hasattr(model, 'cpu'):
                    model.cpu()
                elif hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                    model.model.cpu()
            
            # Nettoyage GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Nettoyage mémoire Python
            gc.collect()
            
            logger.info("🧹 Mémoire nettoyée pour traitement de documents longs")
            
        except Exception as e:
            logger.warning(f"⚠️ Nettoyage mémoire partiel: {e}")

    def get_model_status(self) -> Dict:
        """Retourne le statut des modèles scientifiques"""
        return {
            "models_loaded": self.models_loaded,
            "device": str(self.device),
            "loaded_models": list(self.loaded_models.keys()),
            "max_context_length": 16384,  # LED
            "supported_languages": list(set([lang for lang in self.language_codes.keys() if len(lang) > 2])),
            "specialized_for": "scientific_articles_long_documents"
        }


# Instance globale spécialisée
scientific_processor = ScientificArticleProcessor()

# Fonction de compatibilité
def get_multilingual_models(device="auto"):
    return ScientificArticleProcessor(device=device)

if __name__ == "__main__":
    # Test avec un texte scientifique long
    processor = ScientificArticleProcessor()
    print("🔬 Statut:", processor.get_model_status())
    
    # Test de résumé scientifique
    scientific_text = """
    Abstract: This comprehensive study investigates the impact of deep learning architectures on medical image analysis. 
    We evaluated convolutional neural networks (CNNs) and transformer-based models across multiple medical imaging modalities.
    
    Introduction: Medical image analysis has undergone significant transformation with the advent of deep learning. 
    Traditional machine learning approaches are increasingly being replaced by sophisticated neural networks.
    
    Methodology: We conducted a systematic review of 250 peer-reviewed studies from 2018 to 2024. 
    The analysis included CT scans, MRI images, and histological samples across various medical conditions.
    
    Results: Our findings demonstrate that transformer-based models achieve 23% higher accuracy in anomaly detection 
    compared to traditional CNNs. However, computational requirements remain a significant challenge.
    
    Discussion: The superior performance of attention-based architectures suggests potential for clinical deployment, 
    though interpretability and computational efficiency require further investigation.
    
    Conclusion: Deep learning continues to revolutionize medical imaging, with transformer models showing particular promise 
    for complex diagnostic tasks. Future work should focus on model optimization and clinical validation.
    """
    
    result = processor.summarize_long_article(
        scientific_text, 
        language="anglais",
        summary_type="structured"
    )
    
    print(f"📊 Métriques: {result['original_metrics']}")
    print(f"📝 Résumé: {result['summary'][:500]}...")
    
    # Test de métadonnées
    metadata = processor.extract_scientific_metadata(scientific_text)
    print(f"🔍 Métadonnées: {metadata}")