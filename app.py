import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from bs4 import BeautifulSoup
import re

# Import des modèles multilingues avancés
from models import multilingual_models

# Configuration de la page
st.set_page_config(
    page_title="Résumé & Traduction d'Articles",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #5078C8;
        margin-bottom: 1rem;
        border-bottom: 2px solid #7ffbff;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #7ffbff;
        margin: 0.5rem 0;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .translation-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #7ffbff;
    }
    .arabic-text {
        text-align: right;
        direction: rtl;
        font-family: 'Arial', sans-serif;
    }
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .extraction-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Couleurs
PRIMARY_COLOR = "#7ffbff"

class SummarizationApp:
    def __init__(self):
        self.init_session_state()
        # Chargement des modèles multilingues
        self.models = multilingual_models
        
    def init_session_state(self):
        """Initialise l'état de la session"""
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        if 'auto_demo' not in st.session_state:
            st.session_state.auto_demo = False
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        if 'extracted_text' not in st.session_state:
            st.session_state.extracted_text = ""
        if 'show_processing_options' not in st.session_state:
            st.session_state.show_processing_options = False
        if 'url_content' not in st.session_state:
            st.session_state.url_content = ""

    def extract_text_from_pdf_url(self, url):
        """Tente d'extraire le texte d'un PDF en ligne"""
        try:
            # Pour les PDFs, on ne peut pas facilement extraire le texte sans bibliothèques spécialisées
            # On retourne un message d'information
            return f"PDF détecté à l'URL: {url}\n\nPour traiter un PDF, veuillez le télécharger et l'uploader via l'option 'Fichier Texte'."
        except Exception as e:
            return f"Erreur avec le PDF: {str(e)}"

    def scrape_web_content(self, url):
        """Extrait le contenu d'une page web avec gestion améliorée"""
        try:
            # Headers pour éviter le blocage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'fr-FR,fr;q=0.8,en-US;q=0.5,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Vérifier si c'est un PDF
            if url.lower().endswith('.pdf'):
                return self.extract_text_from_pdf_url(url)
            
            # Faire la requête
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Vérifier le type de contenu
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type:
                return self.extract_text_from_pdf_url(url)
            
            # Parser le HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Supprimer les éléments indésirables
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Essayer différentes stratégies d'extraction
            text_parts = []
            
            # Stratégie 1: Contenu principal avec des sélecteurs communs
            main_selectors = [
                'article',
                'main',
                '[role="main"]',
                '.content',
                '.main-content',
                '.post-content',
                '.article-content',
                '.entry-content'
            ]
            
            for selector in main_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if len(text) > 100:  # Un contenu significatif
                        text_parts.append(text)
            
            # Stratégie 2: Tous les paragraphes si la stratégie 1 échoue
            if not text_parts:
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 50:  # Paragraphes significatifs
                        text_parts.append(text)
            
            # Stratégie 3: Titres et sous-titres
            if not text_parts:
                headings = soup.find_all(['h1', 'h2', 'h3'])
                for heading in headings:
                    text = heading.get_text(strip=True)
                    if len(text) > 20:
                        text_parts.append(text)
            
            # Nettoyer et combiner le texte
            if text_parts:
                full_text = '\n\n'.join(text_parts)
                # Nettoyer les espaces multiples
                full_text = re.sub(r'\s+', ' ', full_text)
                full_text = full_text.strip()
                
                if len(full_text) > 100:
                    return full_text
                else:
                    return "Le contenu extrait est trop court. Le site peut être protégé ou utiliser du contenu dynamique."
            else:
                return "Aucun contenu textuel significatif n'a pu être extrait. Le site peut utiliser du JavaScript pour afficher le contenu."
                
        except requests.exceptions.RequestException as e:
            return f"Erreur de connexion: {str(e)}"
        except Exception as e:
            return f"Erreur lors de l'extraction: {str(e)}"

    def process_with_models(self, text, source_lang, target_langs, summary_length):
        """Utilise les modèles avancés pour le résumé et la traduction"""
        try:
            # Étape 1: Résumé du texte
            summary = self.models.summarize_text(text, source_lang, summary_length)
            
            # Étape 2: Traductions multiples
            translations = {}
            for target_lang in target_langs:
                if target_lang != source_lang:  # Éviter la traduction vers la même langue
                    translation = self.models.translate_text(summary, source_lang, target_lang)
                    translations[target_lang] = translation
            
            # Métriques
            metrics = {
                "original_length": len(text.split()),
                "summary_length": len(summary.split()),
                "reduction_percentage": round((1 - len(summary.split())/len(text.split())) * 100, 1),
                "processing_time": 0,
                "translations_count": len(translations)
            }
            
            return {
                "summary": summary,
                "translations": translations,
                "metrics": metrics,
                "source_lang": source_lang
            }
            
        except Exception as e:
            st.error(f"Erreur lors du traitement: {str(e)}")
            return None

    def home_section(self):
        """Section d'accueil"""
        st.markdown('<div class="main-header">SummarEase Pro</div>', unsafe_allow_html=True)
        
        # Afficher directement le titre
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>Bienvenue dans SummarEase Pro</h1>
            <p style='font-size: 1.2rem;'>Résumé & Traduction Multilingue Avancés</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statut des modèles
        model_status = self.models.get_model_status()
        if model_status["models_loaded"]:
            st.success("✅ **Modèles avancés chargés avec succès**")
            
            # Afficher les langues supportées
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 🌍 Langues Supportées")
                st.write("• Français")
                st.write("• Anglais")
                st.write("• Espagnol")
                st.write("• Allemand")
                st.write("• Arabe")
            with col2:
                st.markdown("### 🔧 Fonctionnalités")
                st.write("• Résumé intelligent")
                st.write("• Traduction multilingue")
                st.write("• Extraction web avancée")
                st.write("• Support fichiers")
            with col3:
                st.markdown("### ⚡ Performances")
                st.write(f"• Device: {model_status['device'].upper()}")
                st.write("• Traitement rapide")
                st.write("• Qualité optimale")

    def input_section(self):
        """Section de chargement des articles"""
        st.markdown('<div class="section-header">📂 Chargement de l\'Article</div>', unsafe_allow_html=True)
        
        # Réinitialiser l'état si nécessaire
        if st.session_state.processing_complete:
            st.session_state.processing_complete = False
            st.session_state.show_processing_options = False

        input_method = st.radio(
            "Méthode de saisie :",
            ["📝 Texte Manuel", "🔗 Lien Web", "📄 Fichier Texte"],
            horizontal=True,
            key="input_method_radio"
        )
        
        extraction_success = False
        
        if input_method == "📝 Texte Manuel":
            input_text = st.text_area(
                "Collez votre article ici :",
                height=200,
                placeholder="Collez le texte de l'article à résumer et traduire...",
                key="manual_text_area"
            )
            if input_text.strip():
                st.session_state.extracted_text = input_text
                extraction_success = True
                st.success("✅ Texte prêt pour le traitement!")
            
        elif input_method == "🔗 Lien Web":
            st.markdown("#### 🌐 Extraction à partir d'une URL")
            url = st.text_input("Entrez l'URL de l'article :", 
                               placeholder="https://example.com/article",
                               key="url_input")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🌐 Extraire le contenu", use_container_width=True, key="extract_button"):
                    if url:
                        with st.spinner("Extraction du contenu web en cours..."):
                            extracted_content = self.scrape_web_content(url)
                            st.session_state.url_content = extracted_content
                            
                            # Vérifier si l'extraction a réussi
                            if not any(error in extracted_content.lower() for error in ["erreur", "aucun contenu", "trop court", "protégé"]):
                                st.session_state.extracted_text = extracted_content
                                extraction_success = True
                                
                                st.markdown('<div class="extraction-success">✅ Contenu extrait avec succès!</div>', unsafe_allow_html=True)
                                
                                # Afficher un aperçu du contenu
                                with st.expander("📄 Aperçu du contenu extrait", expanded=True):
                                    word_count = len(extracted_content.split())
                                    st.metric("Nombre de mots", word_count)
                                    st.text_area("Contenu:", 
                                               extracted_content[:1500] + "..." if len(extracted_content) > 1500 else extracted_content, 
                                               height=200, 
                                               key="preview_area",
                                               label_visibility="collapsed")
                            else:
                                st.error(f"❌ {extracted_content}")
                    else:
                        st.warning("⚠️ Veuillez entrer une URL valide")
            
            # Afficher les conseils d'extraction
            with st.expander("💡 Conseils pour l'extraction web"):
                st.markdown("""
                **Pour de meilleurs résultats:**
                - Utilisez des URLs d'articles de blog ou de news
                - Évitez les sites nécessitant une connexion
                - Les sites avec beaucoup de JavaScript peuvent ne pas fonctionner
                - Les PDFs en ligne ne sont pas supportés
                
                **Sites recommandés:**
                - Wikipedia.org
                - Blog posts
                - Articles de presse
                - Documentation technique
                """)
            
        else:  # Fichier Texte
            st.markdown("#### 📄 Upload de fichier")
            uploaded_file = st.file_uploader("Téléchargez un fichier texte", 
                                           type=['txt'], 
                                           key="file_uploader",
                                           help="Supporte les fichiers .txt")
            if uploaded_file:
                try:
                    if uploaded_file.type == "text/plain":
                        input_text = uploaded_file.getvalue().decode("utf-8")
                        st.session_state.extracted_text = input_text
                        extraction_success = True
                        
                        st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès!")
                        
                        # Afficher un aperçu
                        with st.expander("📄 Aperçu du fichier"):
                            word_count = len(input_text.split())
                            st.metric("Nombre de mots", word_count)
                            st.text_area("Contenu:", 
                                       input_text[:1500] + "..." if len(input_text) > 1500 else input_text, 
                                       height=200,
                                       label_visibility="collapsed")
                            
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
        
        # Afficher les options de traitement si du texte est disponible
        if st.session_state.extracted_text and extraction_success:
            st.session_state.show_processing_options = True
        
        if st.session_state.show_processing_options:
            self.show_processing_options()
        
        return None, None, None, None

    def show_processing_options(self):
        """Affiche les options de traitement après chargement du texte"""
        st.markdown("---")
        st.markdown('<div class="section-header">⚙️ Options de Traitement</div>', unsafe_allow_html=True)
        
        # Configuration du traitement
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_lang = st.selectbox(
                "Langue source :",
                ["français", "anglais", "espagnol", "allemand", "arabe"],
                index=0,
                help="Sélectionnez la langue du texte original",
                key="source_lang_select"
            )
        
        with col2:
            target_langs = st.multiselect(
                "Langues cibles :",
                ["français", "anglais", "espagnol", "allemand", "arabe"],
                default=["anglais", "espagnol", "allemand"],
                help="Sélectionnez une ou plusieurs langues pour la traduction",
                key="target_langs_multiselect"
            )
        
        with col3:
            summary_length = st.select_slider(
                "Longueur du résumé :",
                options=["court", "moyen", "long"],
                value="moyen",
                key="summary_length_slider"
            )
        
        # Boutons d'action
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Lancer le Résumé et la Traduction", 
                        type="primary", 
                        use_container_width=True,
                        key="process_main_button"):
                
                if not target_langs:
                    st.error("❌ Veuillez sélectionner au moins une langue cible")
                    return
                
                if len(st.session_state.extracted_text.strip()) < 50:
                    st.error("❌ Le texte est trop court pour être traité (minimum 50 caractères)")
                    return
                
                # Lancer le traitement
                with st.spinner("Traitement en cours..."):
                    result = self.processing_section(
                        st.session_state.extracted_text, 
                        source_lang, 
                        target_langs, 
                        summary_length
                    )
                
                if result:
                    st.session_state.current_result = result
                    st.session_state.current_section = "📊 Résultats"
                    st.rerun()
                else:
                    st.error("❌ Le traitement a échoué. Veuillez réessayer avec un texte différent.")

    def processing_section(self, input_text, source_lang, target_langs, summary_length):
        """Section de traitement avec les modèles avancés"""
        try:
            # Traitement avec les modèles
            result = self.process_with_models(input_text, source_lang, target_langs, summary_length)
            
            if result:
                st.session_state.current_result = result
                st.session_state.history.append({
                    "timestamp": datetime.now(),
                    "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                    "result": result
                })
                st.session_state.processing_complete = True
                return result
            
            return None
            
        except Exception as e:
            st.error(f"Erreur lors du traitement: {str(e)}")
            return None

    def results_section(self):
        """Section d'affichage des résultats multilingues"""
        if not st.session_state.current_result:
            st.info("ℹ️ Aucun résultat à afficher. Veuillez d'abord traiter un article.")
            st.markdown("""
            <div style='text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px;'>
                <h3 style='color: #666;'>📝 En attente de texte</h3>
                <p>Utilisez l'onglet "📂 Charger Article" pour traiter votre premier texte.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        result = st.session_state.current_result
        
        # Métriques principales
        st.markdown("#### 📊 Métriques de Performance")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Longueur originale", f"{result['metrics']['original_length']} mots")
        
        with col2:
            st.metric("Longueur résumé", f"{result['metrics']['summary_length']} mots")
        
        with col3:
            st.metric("Réduction", f"{result['metrics']['reduction_percentage']}%")
        
        with col4:
            st.metric("Traductions", f"{result['metrics']['translations_count']} langues")
        
        # Résumé original
        st.markdown("#### 📄 Résumé Généré")
        st.markdown(f'<div class="result-card">{result["summary"]}</div>', unsafe_allow_html=True)
        
        # Traductions multilingues
        st.markdown("#### 🌐 Traductions Multilingues")
        
        if result['translations']:
            # Organiser les traductions en colonnes
            translations = list(result['translations'].items())
            num_cols = 2
            
            for i in range(0, len(translations), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(translations):
                        lang, text = translations[i + j]
                        with cols[j]:
                            # Style spécial pour l'arabe
                            if lang == "arabe":
                                st.markdown(
                                    f'<div class="translation-card arabic-text">'
                                    f'<strong>🌍 {lang.capitalize()}</strong><br>{text}'
                                    f'</div>', 
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div class="translation-card">'
                                    f'<strong>🌍 {lang.capitalize()}</strong><br>{text}'
                                    f'</div>', 
                                    unsafe_allow_html=True
                                )
        
        # Actions supplémentaires
        st.markdown("#### 💾 Export des Résultats")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Télécharger JSON", use_container_width=True):
                self.download_json(result)
        
        with col2:
            if st.button("📊 Télécharger CSV", use_container_width=True):
                self.download_csv(result)
        
        with col3:
            if st.button("🔄 Nouveau traitement", use_container_width=True):
                st.session_state.current_result = None
                st.session_state.extracted_text = ""
                st.session_state.show_processing_options = False
                st.rerun()

    def download_json(self, result):
        """Télécharge les résultats en JSON"""
        json_str = json.dumps(result, ensure_ascii=False, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="resultats_multilingues.json">Télécharger JSON</a>'
        st.markdown(href, unsafe_allow_html=True)

    def download_csv(self, result):
        """Télécharge les métriques en CSV"""
        # Préparer les données pour CSV
        data = {
            'original_length': [result['metrics']['original_length']],
            'summary_length': [result['metrics']['summary_length']],
            'reduction_percentage': [result['metrics']['reduction_percentage']],
            'translations_count': [result['metrics']['translations_count']]
        }
        
        # Ajouter les traductions
        for lang, text in result['translations'].items():
            data[f'traduction_{lang}'] = [text]
        
        metrics_df = pd.DataFrame(data)
        csv = metrics_df.to_csv(index=False, encoding='utf-8')
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="resultats_multilingues.csv">Télécharger CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    def run(self):
        """Exécute l'application principale"""
        
        # Menu latéral
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #5078C8;'>SummarEase Pro</h2>
                <p>Résumé & Traduction Multilingue</p>
            </div>
            """, unsafe_allow_html=True)
            
            section = st.radio(
                "Navigation :",
                [
                    "🏠 Accueil",
                    "📂 Charger Article", 
                    "📊 Résultats",
                    "🕒 Historique",
                    "🔧 Info Modèles"
                ],
                key="navigation_radio"
            )
            
            st.markdown("---")
            st.markdown("### 📈 Statut")
            
            # Statut des modèles
            model_status = self.models.get_model_status()
            if model_status["models_loaded"]:
                st.success("✅ Modèles chargés")
                if st.session_state.current_result:
                    st.metric("Dernière réduction", f"{st.session_state.current_result['metrics']['reduction_percentage']}%")
                    st.metric("Traductions", f"{st.session_state.current_result['metrics']['translations_count']}")
                else:
                    st.info("🔄 Prêt pour le traitement")
            else:
                st.error("❌ Modèles non chargés")
            
            # Aperçu du texte chargé
            if st.session_state.extracted_text:
                st.markdown("---")
                st.markdown("### 📝 Texte Chargé")
                word_count = len(st.session_state.extracted_text.split())
                st.info(f"📄 {word_count} mots")
            
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666; font-size: 0.8rem;'>
                Support: Français, Anglais, Espagnol, Allemand, Arabe
            </div>
            """, unsafe_allow_html=True)
        
        # Contenu principal selon la section
        if section == "🏠 Accueil":
            self.home_section()
            
        elif section == "📂 Charger Article":
            self.input_section()
                
        elif section == "📊 Résultats":
            self.results_section()
            
        elif section == "🕒 Historique":
            self.history_section()
            
        elif section == "🔧 Info Modèles":
            self.model_info_section()

    def history_section(self):
        """Section historique"""
        st.markdown('<div class="section-header">🕒 Historique des Traitements</div>', unsafe_allow_html=True)
        
        if not st.session_state.history:
            st.info("ℹ️ Aucun historique disponible.")
            return
        
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"📄 Traitement du {entry['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Entrée :**")
                    st.write(entry['input'])
                    st.write("**Métriques :**")
                    st.metric("Réduction", f"{entry['result']['metrics']['reduction_percentage']}%")
                    st.metric("Traductions", f"{entry['result']['metrics']['translations_count']}")
                with col2:
                    st.write("**Résumé :**")
                    st.write(entry['result']['summary'][:200] + "...")
                    if st.button("🔍 Voir détails", key=f"view_{i}"):
                        st.session_state.current_result = entry['result']
                        st.session_state.current_section = "📊 Résultats"
                        st.rerun()

    def model_info_section(self):
        """Section d'information sur les modèles"""
        st.markdown('<div class="section-header">🔧 Informations des Modèles</div>', unsafe_allow_html=True)
        
        model_status = self.models.get_model_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🏗️ Architecture des Modèles")
            if model_status["models_loaded"]:
                st.success("✅ Modèles chargés avec succès")
                st.write(f"**Device:** {model_status['device']}")
            st.markdown("""
            **Technologies utilisées:**
            - Transformers (Hugging Face)
            - Modèles spécialisés par langue
            - Traduction neuronale
            - Résumé abstractif
            """)
        with col2:
            st.markdown("### 🌍 Langues Supportées")
            main_languages = ["français", "anglais", "espagnol", "allemand", "arabe"]
            for lang in main_languages:
                st.write(f"• ✅ {lang.capitalize()}")
            st.markdown("""
            **Fonctionnalités:**
            - Résumé intelligent
            - Traduction multilingue
            - Extraction web avancée
            - Interface adaptative
            """)

# Lancement de l'application
if __name__ == "__main__":
    app = SummarizationApp()
    app.run()