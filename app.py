import streamlit as st
import requests
import json
import time
import pdfplumber
import pandas as pd
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    class DummyPlotly:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    go = DummyPlotly()
    px = DummyPlotly()
from datetime import datetime
import base64
from bs4 import BeautifulSoup
import re
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import des modèles scientifiques
from models import get_multilingual_models

@st.cache_resource
def load_models():
    return get_multilingual_models()

# Configuration de la page pour documents longs
st.set_page_config(
    page_title="Scientific Article Summarizer Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour interface scientifique
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.6rem;
        color: #2E86AB;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .scientific-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 5px solid #2E86AB;
        border-right: 1px solid #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .processing-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #2E86AB;
        margin: 1rem 0;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-card {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .arabic-text {
        text-align: right;
        direction: rtl;
        font-family: 'Arial', sans-serif;
        line-height: 2;
    }
    .scientific-button {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem 0;
    }
    .scientific-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.4);
    }
    .sidebar-scientific {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .file-upload-box {
        border: 2px dashed #2E86AB;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #f8f9fa;
    }
    .domain-tag {
        background: #2E86AB;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class ScientificSummarizationApp:
    def __init__(self):
        self.init_session_state()
        with st.spinner("🔬 Chargement des modèles scientifiques pour documents longs..."):
            try:
                self.models = load_models()
                if self.models.models_loaded:
                    st.success("✅ **Système scientifique chargé avec succès**")
                    st.info("🎯 **Optimisé pour: Articles scientifiques, Recherches, Documents longs (1-90 pages)**")
                else:
                    st.error("❌ Erreur lors du chargement des modèles scientifiques")
            except Exception as e:
                st.error(f"❌ Erreur critique: {e}")
                logger.error(f"Erreur d'initialisation: {e}")

    def init_session_state(self):
        """Initialise l'état de la session pour documents scientifiques"""
        default_states = {
            'scientific_history': [],
            'current_scientific_result': None,
            'extracted_scientific_text': "",
            'processing_step': 0,
            'document_metadata': None,
            'current_document_type': None,
            'chunk_progress': 0
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def home_section(self):
        """Section d'accueil scientifique"""
        st.markdown('<div class="main-header">🔬 Scientific Article Summarizer Pro</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%); border-radius: 20px; color: white; margin-bottom: 2rem;'>
            <h1 style='font-size: 2.8rem; margin-bottom: 1.5rem;'>Résumé & Traduction Scientifique Avancé</h1>
            <p style='font-size: 1.4rem; opacity: 0.9;'>Spécialisé pour les articles de recherche, thèses et documents académiques longs</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Capacités du système
        model_status = self.models.get_model_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📚 Types de Documents")
            st.write("• Articles de recherche")
            st.write("• Thèses et mémoires")
            st.write("• Documents académiques")
            st.write("• Publications scientifiques")
            st.write("• Rapports techniques")
        
        with col2:
            st.markdown("### 🔧 Capacités Techniques")
            st.write(f"• Contexte: {model_status['max_context_length']} tokens")
            st.write("• Traitement hiérarchique")
            st.write("• Découpage intelligent")
            st.write("• Métadonnées scientifiques")
            st.write("• Multilingue avancé")
        
        with col3:
            st.markdown("### 🌍 Domaines Supportés")
            st.write("• Médecine/Biologie")
            st.write("• Informatique/IA")
            st.write("• Physique/Chimie")
            st.write("• Mathématiques")
            st.write("• Sciences Sociales")

        # Métriques système
        st.markdown("---")
        st.markdown("### 📊 Statut du Système Scientifique")
        
        status_cols = st.columns(4)
        with status_cols[0]:
            st.metric("Modèles Chargés", len(model_status['loaded_models']))
        with status_cols[1]:
            st.metric("Device", model_status['device'].upper())
        with status_cols[2]:
            st.metric("Contexte Max", "16K tokens")
        with status_cols[3]:
            st.metric("Langues", len(model_status['supported_languages']))

    def input_section(self):
        """Section de chargement pour documents scientifiques"""
        st.markdown('<div class="section-header">📂 Chargement du Document Scientifique</div>', unsafe_allow_html=True)
        
        # Réinitialisation si nécessaire
        if st.session_state.current_scientific_result:
            st.session_state.current_scientific_result = None

        input_method = st.radio(
            "Méthode de saisie :",
            ["📝 Texte Direct", "🔗 URL Scientifique", "📄 PDF Académique", "📁 Fichier Texte"],
            horizontal=True,
            key="scientific_input_method"
        )

        extraction_success = False
        
        if input_method == "📝 Texte Direct":
            st.markdown("#### 🎯 Collage du Document Scientifique")
            input_text = st.text_area(
                "Collez votre article scientifique complet :",
                height=300,
                placeholder="Collez ici le texte complet de votre article de recherche, thèse, ou document académique...",
                key="scientific_text_area",
                help="Supporte jusqu'à 90 pages de texte (environ 45,000 mots)"
            )
            if input_text.strip():
                word_count = len(input_text.split())
                if word_count >= 100:
                    st.session_state.extracted_scientific_text = input_text
                    extraction_success = True
                    st.success(f"✅ Document prêt! ({word_count} mots, ~{max(1, word_count//500)} pages)")
                else:
                    st.warning(f"⚠️ Texte trop court pour un document scientifique ({word_count} mots)")
            
        elif input_method == "🔗 URL Scientifique":
            st.markdown("#### 🌐 Extraction depuis une URL Scientifique")
            url = st.text_input("Entrez l'URL de l'article scientifique :", 
                               placeholder="https://arxiv.org/abs/... ou https://www.ncbi.nlm.nih.gov/pmc/articles/...",
                               key="scientific_url_input")
            
            if st.button("🔍 Extraire le Contenu Scientifique", use_container_width=True, key="extract_scientific"):
                if url:
                    with st.spinner("🔬 Extraction du contenu scientifique..."):
                        extracted_content = self.models.scrape_web_content(url)
                        
                        if not any(error in extracted_content.lower() for error in ["erreur", "insuffisant", "trop court"]):
                            word_count = len(extracted_content.split())
                            if word_count >= 200:
                                st.session_state.extracted_scientific_text = extracted_content
                                extraction_success = True
                                
                                with st.expander("📊 Aperçu du Document Extrait", expanded=True):
                                    st.metric("Mots extraits", word_count)
                                    st.metric("Pages estimées", max(1, word_count//500))
                                    st.text_area("Extrait:", 
                                               extracted_content[:2000] + "..." if len(extracted_content) > 2000 else extracted_content, 
                                               height=250, 
                                               key="scientific_preview",
                                               label_visibility="collapsed")
                            else:
                                st.error("❌ Contenu scientifique insuffisant extrait")
                        else:
                            st.error(f"❌ {extracted_content}")
                else:
                    st.warning("⚠️ Veuillez entrer une URL valide")
            
        elif input_method == "📄 PDF Académique":
            st.markdown("#### 📄 Upload de PDF Scientifique")
            st.markdown('<div class="file-upload-box">', unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader(
                "Déposez votre PDF scientifique ici",
                type=['pdf'],
                key="scientific_pdf_uploader",
                help="Supporte les PDFs de recherche, thèses, articles (max 90 pages)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if uploaded_pdf:
                try:
                    with st.spinner("📖 Extraction du texte du PDF..."):
                        text_content = ""
                        with pdfplumber.open(uploaded_pdf) as pdf:
                            total_pages = len(pdf.pages)
                            progress_bar = st.progress(0)
                            
                            for i, page in enumerate(pdf.pages):
                                extracted = page.extract_text()
                                if extracted:
                                    text_content += extracted + "\n\n"
                                progress_bar.progress((i + 1) / total_pages)

                        word_count = len(text_content.split())
                        
                        if word_count >= 200:
                            st.session_state.extracted_scientific_text = text_content
                            extraction_success = True
                            
                            st.success(f"✅ PDF '{uploaded_pdf.name}' extrait avec succès!")
                            st.info(f"📊 Document: {word_count} mots, {total_pages} pages")

                            with st.expander("🔍 Aperçu du PDF", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Pages", total_pages)
                                    st.metric("Mots", word_count)
                                with col2:
                                    st.metric("Caractères", len(text_content))
                                    st.metric("Pages estimées", max(1, word_count//500))
                                
                                st.text_area("Contenu extrait :",
                                           text_content[:2500] + "..." if len(text_content) > 2500 else text_content,
                                           height=300,
                                           label_visibility="collapsed")

                        else:
                            st.error("❌ Le PDF ne contient pas assez de texte lisible")

                except Exception as e:
                    st.error(f"❌ Erreur lors de la lecture du PDF : {str(e)}")
        
        else:  # Fichier Texte
            st.markdown("#### 📁 Upload de Fichier Texte")
            uploaded_file = st.file_uploader("Téléchargez votre document texte", 
                                           type=['txt', 'docx'], 
                                           key="scientific_file_uploader",
                                           help="Supporte .txt et .docx")
            if uploaded_file:
                try:
                    if uploaded_file.type == "text/plain":
                        input_text = uploaded_file.getvalue().decode("utf-8")
                    else:
                        # Pour .docx, on utiliserait python-docx, mais pour simplifier:
                        input_text = "Contenu DOCX - Veuillez utiliser le format PDF ou texte simple."
                    
                    word_count = len(input_text.split())
                    if word_count >= 100:
                        st.session_state.extracted_scientific_text = input_text
                        extraction_success = True
                        st.success(f"✅ Fichier scientifique chargé! ({word_count} mots)")
                    else:
                        st.warning("⚠️ Le fichier contient moins de 100 mots")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement: {str(e)}")
        
        # Afficher les options de traitement scientifique si du texte est disponible
        if st.session_state.extracted_scientific_text and extraction_success:
            self.show_scientific_processing_options()
        
        return None

    def show_scientific_processing_options(self):
        """Affiche les options de traitement scientifique"""
        st.markdown("---")
        st.markdown('<div class="section-header">⚗️ Options de Traitement Scientifique</div>', unsafe_allow_html=True)
        
        # Analyse préliminaire du document
        with st.spinner("🔍 Analyse du document..."):
            metadata = self.models.extract_scientific_metadata(st.session_state.extracted_scientific_text)
            st.session_state.document_metadata = metadata
        
        # Affichage des métadonnées
        st.markdown("#### 📋 Analyse du Document")
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.metric("Mots", metadata['word_count'])
        with meta_cols[1]:
            st.metric("Pages estimées", metadata['estimated_pages'])
        with meta_cols[2]:
            st.metric("Langue", metadata['language'])
        with meta_cols[3]:
            st.metric("Sections", len(metadata['sections_present']))
        
        # Domaines détectés
        if metadata['domains']:
            st.write("**Domaines détectés:**")
            for domain in metadata['domains']:
                st.markdown(f'<span class="domain-tag">{domain}</span>', unsafe_allow_html=True)
        
        # Configuration du traitement
        st.markdown("#### ⚙️ Configuration du Traitement")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_lang = st.selectbox(
                "Langue source :",
                ["anglais", "français", "espagnol", "allemand", "arabe"],
                index=0,
                help="Langue du document original",
                key="scientific_source_lang"
            )
        
        with col2:
            target_langs = st.multiselect(
                "Langues de traduction :",
                ["français", "anglais", "espagnol", "allemand", "arabe"],
                default=["français", "anglais"],
                help="Traduire le résumé dans ces langues",
                key="scientific_target_langs"
            )
        
        with col3:
            summary_type = st.selectbox(
                "Type de résumé :",
                ["structured", "abstract", "key_points", "comprehensive"],
                format_func=lambda x: {
                    "structured": "📊 Structuré (Recommandé)",
                    "abstract": "📋 Abstract",
                    "key_points": "🎯 Points Clés", 
                    "comprehensive": "🔍 Complet"
                }[x],
                help="Format du résumé généré",
                key="scientific_summary_type"
            )
        
        # Avertissement pour documents très longs
        if metadata['word_count'] > 10000:
            st.markdown("""
            <div class="warning-card">
                <strong>⚠️ Document Très Long Détecté</strong><br>
                Ce document contient plus de 10,000 mots. Le traitement peut prendre plusieurs minutes.
                Le système utilisera une stratégie de découpage hiérarchique pour maintenir la qualité.
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton de traitement principal
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Lancer l'Analyse Scientifique", 
                        type="primary", 
                        use_container_width=True,
                        key="scientific_process_button"):
                
                if not target_langs:
                    st.error("❌ Veuillez sélectionner au moins une langue de traduction")
                    return
                
                # Validation finale
                if metadata['word_count'] < 100:
                    st.error("❌ Le document est trop court pour une analyse scientifique")
                    return
                
                # Lancer le traitement scientifique
                result = self.scientific_processing_section(
                    st.session_state.extracted_scientific_text, 
                    source_lang, 
                    target_langs, 
                    summary_type
                )
                
                if result:
                    st.session_state.current_scientific_result = result
                    st.rerun()
                else:
                    st.error("❌ L'analyse scientifique a échoué")

    def scientific_processing_section(self, text: str, source_lang: str, target_langs: List[str], summary_type: str):
        """Traitement scientifique avec progression détaillée"""
        try:
            # Interface de progression
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.markdown("#### 🔬 Progression de l'Analyse Scientifique")
                
                # Barres de progression multiples
                main_progress = st.progress(0)
                chunk_progress = st.progress(0)
                status_text = st.empty()
                
                steps = [
                    "📖 Préparation du document...",
                    "🔍 Extraction des métadonnées...", 
                    "📝 Génération du résumé...",
                    "🌍 Traductions scientifiques...",
                    "📊 Finalisation..."
                ]
            
            # Étape 1: Préparation
            status_text.text(steps[0])
            main_progress.progress(10)
            time.sleep(0.5)
            
            # Étape 2: Résumé scientifique
            status_text.text(steps[2])
            main_progress.progress(30)
            
            with st.spinner("🎯 Génération du résumé scientifique..."):
                result = self.models.summarize_long_article(text, source_lang, summary_type)
                main_progress.progress(60)
            
            # Étape 3: Traductions
            status_text.text(steps[3])
            translation_count = len([lang for lang in target_langs if lang != source_lang])
            
            if translation_count > 0:
                translations = {}
                for i, target_lang in enumerate(target_langs):
                    if target_lang != source_lang:
                        status_text.text(f"🌍 Traduction scientifique en {target_lang} ({i+1}/{translation_count})...")
                        try:
                            translation = self.models.translate_scientific_text(
                                result["summary"], source_lang, target_lang
                            )
                            translations[target_lang] = translation
                            
                            # Mise à jour de la progression
                            progress = 60 + (i / translation_count) * 30
                            main_progress.progress(int(progress))
                            
                        except Exception as e:
                            st.warning(f"⚠️ Traduction {source_lang}→{target_lang} échouée: {str(e)}")
                            translations[target_lang] = f"[Erreur: {str(e)}]"
                
                result["translations"] = translations
            
            # Finalisation
            status_text.text(steps[4])
            main_progress.progress(95)
            
            # Ajout des métadonnées
            result["metadata"] = st.session_state.document_metadata
            result["processing_timestamp"] = datetime.now()
            
            main_progress.progress(100)
            status_text.text("✅ Analyse scientifique terminée!")
            
            time.sleep(1)
            
            # Ajout à l'historique
            st.session_state.scientific_history.append({
                "timestamp": datetime.now(),
                "metadata": st.session_state.document_metadata,
                "result": result
            })
            
            return result
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement scientifique: {str(e)}")
            logger.error(f"Scientific processing error: {str(e)}")
            return None
        finally:
            # Nettoyage mémoire
            self.models.cleanup_memory()

    def results_section(self):
        """Section des résultats scientifiques"""
        if not st.session_state.current_scientific_result:
            st.info("""
            ### 🔬 En attente de document scientifique
            Utilisez l'onglet **"📂 Charger Document"** pour analyser votre premier article de recherche.
            
            **Documents supportés:**
            - Articles de recherche (PDF, texte)
            - Thèses et mémoires  
            - Publications académiques
            - Documents techniques longs (1-90 pages)
            """)
            return
        
        result = st.session_state.current_scientific_result
        
        # En-tête des résultats
        st.markdown("### 📊 Résultats de l'Analyse Scientifique")
        
        # Métriques détaillées
        st.markdown("#### 📈 Métriques du Document")
        meta_cols = st.columns(5)
        
        with meta_cols[0]:
            st.metric("Mots originaux", result["metadata"]["word_count"])
        with meta_cols[1]:
            st.metric("Mots résumé", len(result["summary"].split()))
        with meta_cols[2]:
            reduction = result["original_metrics"]["reduction_percentage"] if "reduction_percentage" in result["original_metrics"] else round((1 - len(result["summary"].split())/result["metadata"]["word_count"]) * 100, 1)
            st.metric("Compression", f"{reduction}%")
        with meta_cols[3]:
            st.metric("Pages estimées", result["metadata"]["estimated_pages"])
        with meta_cols[4]:
            st.metric("Stratégie", result.get("processing_strategy", "standard"))
        
        # Domaines et mots-clés
        if result["metadata"]["domains"] or result["metadata"]["keywords"]:
            st.markdown("#### 🔍 Métadonnées Scientifiques")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Domaines détectés:**")
                for domain in result["metadata"]["domains"]:
                    st.markdown(f'<span class="domain-tag">{domain}</span>', unsafe_allow_html=True)
            
            with col2:
                st.write("**Mots-clés principaux:**")
                for keyword in result["metadata"]["keywords"][:8]:
                    st.write(f"• {keyword}")
        
        # Résumé scientifique
        st.markdown("#### 📄 Résumé Scientifique Généré")
        st.markdown(f'<div class="scientific-card">{result["summary"]}</div>', unsafe_allow_html=True)
        
        # Traductions scientifiques
        if "translations" in result and result["translations"]:
            st.markdown("#### 🌐 Traductions Scientifiques")
            translations = list(result["translations"].items())
            
            for lang, text in translations:
                with st.expander(f"🌍 {lang.capitalize()}", expanded=(lang == "français")):
                    if lang == "arabe":
                        st.markdown(f'<div class="scientific-card arabic-text">{text}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="scientific-card">{text}</div>', unsafe_allow_html=True)
        
        # Actions d'export
        st.markdown("#### 💾 Export des Résultats")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            if st.button("📥 Télécharger Rapport Complet", use_container_width=True):
                self.download_scientific_report(result)
        
        with exp_col2:
            if st.button("🔢 Exporter Métadonnées", use_container_width=True):
                self.export_scientific_metadata(result)
        
        with exp_col3:
            if st.button("🔄 Nouvelle Analyse", use_container_width=True):
                self.reset_scientific_analysis()

    def download_scientific_report(self, result):
        """Télécharge un rapport scientifique complet"""
        report = {
            "scientific_analysis_report": {
                "timestamp": result["processing_timestamp"].isoformat() if "processing_timestamp" in result else datetime.now().isoformat(),
                "document_metrics": result["metadata"],
                "summary": result["summary"],
                "translations": result.get("translations", {}),
                "processing_strategy": result.get("processing_strategy", "standard"),
                "sections_analyzed": result.get("sections_analyzed", [])
            }
        }
        
        json_str = json.dumps(report, ensure_ascii=False, indent=2, default=str)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="rapport_scientifique.json">📥 Télécharger le rapport JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

    def export_scientific_metadata(self, result):
        """Exporte les métadonnées scientifiques"""
        metadata_df = pd.DataFrame([{
            'domain': ', '.join(result["metadata"]["domains"]),
            'keywords': ', '.join(result["metadata"]["keywords"][:10]),
            'word_count': result["metadata"]["word_count"],
            'pages': result["metadata"]["estimated_pages"],
            'language': result["metadata"]["language"],
            'sections': len(result["metadata"]["sections_present"]),
            'compression_rate': f"{round((1 - len(result['summary'].split())/result['metadata']['word_count']) * 100, 1)}%"
        }])
        
        csv = metadata_df.to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="metadonnees_scientifiques.csv">📊 Exporter les métadonnées CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    def reset_scientific_analysis(self):
        """Réinitialise l'analyse scientifique"""
        self.models.cleanup_memory()
        st.session_state.current_scientific_result = None
        st.session_state.extracted_scientific_text = ""
        st.session_state.document_metadata = None
        st.rerun()

    def history_section(self):
        """Section historique des analyses scientifiques"""
        st.markdown('<div class="section-header">🕒 Historique des Analyses Scientifiques</div>', unsafe_allow_html=True)
        
        if not st.session_state.scientific_history:
            st.info("""
            ### 📚 Aucune analyse enregistrée
            Les analyses scientifiques que vous effectuerez seront sauvegardées ici pour référence future.
            
            **Chaque analyse conserve:**
            - Les métadonnées du document
            - Le résumé généré
            - Les traductions
            - La stratégie de traitement utilisée
            """)
            return
        
        # Affichage de l'historique
        for i, entry in enumerate(reversed(st.session_state.scientific_history[-10:])):
            with st.expander(f"🔬 Analyse du {entry['timestamp'].strftime('%d/%m/%Y à %H:%M')}", expanded=(i==0)):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**📊 Métriques:**")
                    st.metric("Mots", entry['metadata']['word_count'])
                    st.metric("Pages", entry['metadata']['estimated_pages'])
                    st.metric("Domaines", len(entry['metadata']['domains']))
                    
                    st.write("**🔍 Domaines:**")
                    for domain in entry['metadata']['domains'][:3]:
                        st.markdown(f'<span class="domain-tag">{domain}</span>', unsafe_allow_html=True)
                
                with col2:
                    st.write("**📄 Résumé (extrait):**")
                    st.write(entry['result']['summary'][:300] + "..." if len(entry['result']['summary']) > 300 else entry['result']['summary'])
                    
                    if st.button("📖 Voir l'analyse complète", key=f"view_full_{i}"):
                        st.session_state.current_scientific_result = entry['result']
                        st.rerun()

    def model_info_section(self):
        """Section d'information sur les modèles scientifiques"""
        st.markdown('<div class="section-header">🔧 Informations des Modèles Scientifiques</div>', unsafe_allow_html=True)
        
        model_status = self.models.get_model_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏗️ Architecture Scientifique")
            if model_status["models_loaded"]:
                st.success("✅ **Système scientifique opérationnel**")
                st.write(f"**Device:** {model_status['device']}")
                st.write(f"**Contexte maximum:** {model_status['max_context_length']} tokens")
                st.write(f"**Modèles actifs:** {len(model_status['loaded_models'])}")
                
                st.markdown("""
                **Optimisations scientifiques:**
                - Traitement hiérarchique des longs documents
                - Découpage intelligent avec chevauchement  
                - Détection automatique des sections
                - Extraction de métadonnées scientifiques
                - Gestion mémoire avancée
                """)
            
        with col2:
            st.markdown("### 📚 Spécialisations")
            st.markdown("""
            **Documents supportés:**
            ✅ Articles de recherche
            ✅ Thèses et mémoires
            ✅ Publications académiques
            ✅ Documents techniques
            ✅ Rapports scientifiques
            
            **Langues scientifiques:**
            🇫🇷 Français
            🇬🇧 Anglais  
            🇪🇸 Espagnol
            🇩🇪 Allemand
            🇦🇪 Arabe
            """)
            
            # Test de performance
            if st.button("🧪 Test de performance scientifique", key="scientific_test"):
                with st.spinner("Exécution du test..."):
                    test_text = "This is a scientific test document " * 50
                    try:
                        start_time = time.time()
                        result = self.models.summarize_long_article(test_text, "anglais", "structured")
                        end_time = time.time()
                        
                        st.success(f"✅ Test réussi en {end_time - start_time:.2f} secondes")
                        st.metric("Performance", f"{end_time - start_time:.2f}s")
                    except Exception as e:
                        st.error(f"❌ Test échoué: {e}")
        
        # Nettoyage mémoire
        st.markdown("---")
        if st.button("🗑️ Nettoyer la mémoire scientifique", use_container_width=True):
            self.models.cleanup_memory()
            st.success("🧹 Mémoire nettoyée pour les prochains documents scientifiques!")

    def run(self):
        """Exécute l'application scientifique"""
        
        # Menu latéral scientifique
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #2E86AB;'>🔬 Scientific Pro</h2>
                <p>Analyse de Documents Scientifiques</p>
            </div>
            """, unsafe_allow_html=True)
            
            section = st.radio(
                "Navigation :",
                [
                    "🏠 Accueil Scientifique",
                    "📂 Charger Document", 
                    "📊 Résultats",
                    "🕒 Historique",
                    "🔧 Info Modèles"
                ],
                key="scientific_navigation"
            )
            
            st.markdown("---")
            st.markdown("### 📈 Statut du Document")
            
            # Statut actuel
            if st.session_state.current_scientific_result:
                result = st.session_state.current_scientific_result
                st.success("✅ Analyse terminée")
                st.metric("Compression", f"{round((1 - len(result['summary'].split())/result['metadata']['word_count']) * 100, 1)}%")
                st.metric("Traductions", len(result.get('translations', {})))
            elif st.session_state.extracted_scientific_text:
                metadata = st.session_state.document_metadata or {}
                st.info("📄 Document chargé")
                st.metric("Mots", metadata.get('word_count', 'N/A'))
                st.metric("Pages", metadata.get('estimated_pages', 'N/A'))
            else:
                st.info("📝 Prêt pour analyse")
            
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666; font-size: 0.8rem;'>
                v3.0 Scientifique • Documents longs (1-90 pages)
            </div>
            """, unsafe_allow_html=True)
        
        # Contenu principal
        if section == "🏠 Accueil Scientifique":
            self.home_section()
        elif section == "📂 Charger Document":
            self.input_section()
        elif section == "📊 Résultats":
            self.results_section()
        elif section == "🕒 Historique":
            self.history_section()
        elif section == "🔧 Info Modèles":
            self.model_info_section()

# Lancement de l'application scientifique
if __name__ == "__main__":
    try:
        app = ScientificSummarizationApp()
        app.run()
    except Exception as e:
        st.error(f"❌ L'application scientifique a rencontré une erreur: {e}")
        logger.error(f"Scientific application crash: {e}")