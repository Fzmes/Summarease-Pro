# SummarEase Pro 🌍

<div align="center">

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-square&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-square&logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

**AI-Powered Multilingual Summarization & Translation Platform**

*Extract web content • Generate intelligent summaries • Translate across 12+ languages*

[Installation](#-installation) • [Features](#-features) • [Usage](#-usage) • [Languages](#-languages)

</div>

## 📖 Overview

SummarEase Pro is an advanced AI-powered web application that provides intelligent text summarization and multilingual translation capabilities. Built with state-of-the-art transformer models, it enables users to extract key information from articles, documents, and web pages, then translate the summaries into multiple languages.

## ✨ Features

### 🤖 AI-Powered Processing
- **Smart Summarization**: Specialized models for different languages
- **Multilingual Translation**: Support for 12+ languages
- **Advanced Models**: BART, mT5, M2M100, and MarianMT transformers

### 🌐 Content Extraction
- **Web Scraping**: Extract content from any URL
- **Multiple Inputs**: Text input, URL scraping, file upload
- **Content Cleaning**: Remove scripts and non-content elements automatically

### 💬 Language Support
- **Core Languages**: French, English, Spanish, German, Arabic
- **Additional Languages**: Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, Hindi
- **Bidirectional Translation**: Translate between any language pair

### 🎯 User Experience
- **Beautiful Interface**: Modern Streamlit dashboard
- **Real-time Processing**: Live progress tracking
- **Export Results**: Download in JSON and CSV formats
- **History Tracking**: Review previous operations

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Fzmes/summarease-pro.git
cd summarease-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch the application**
```bash
streamlit run app.py
```

4. **Open your browser**
   - Application opens automatically at `http://localhost:8501`
   - Or manually navigate to the displayed URL

## 📸 How to Use

### Process Text Directly
1. Go to "📂 Charger Article" section
2. Select "📝 Texte Manuel"
3. Paste your text and configure options
4. Click "🚀 Lancer le Résumé et la Traduction"

### Extract Web Content
1. Select "🔗 Lien Web" 
2. Enter article URL (Wikipedia, news sites, blogs)
3. Click "🌐 Extraire le contenu"
4. Process the extracted content

### Upload Files
1. Choose "📄 Fichier Texte"
2. Upload .txt files
3. View preview and process

## 🗣️ Supported Languages

| Language | Code | Summarization | Translation |
|----------|------|---------------|-------------|
| Français | `fr` | ✅ Specialized | ✅ All pairs |
| English | `en` | ✅ Specialized | ✅ All pairs |
| Español | `es` | ✅ Available | ✅ All pairs |
| Deutsch | `de` | ✅ Available | ✅ All pairs |
| العربية | `ar` | ✅ Available | ✅ All pairs |
| 中文 | `zh` | ✅ Available | ✅ All pairs |
| 日本語 | `ja` | ✅ Available | ✅ All pairs |

## 🏗️ Technical Architecture

### AI Models Used
```
Summarization:
├── French: moussaKam/barthez-orangesum-abstract
├── English: facebook/bart-large-cnn
└── Multilingual: google/mt5-small

Translation:
├── M2M100: facebook/m2m100_418M (100 languages)
├── MBART-50: facebook/mbart-large-50-many-to-many-mmt
└── MarianMT: Helsinki-NLP specialized models
```

### Core Components
- **ExtendedMultilingualModels**: Manages AI models and processing
- **SummarizationApp**: Streamlit web interface controller
- **Web Scraper**: Extracts content from websites
- **Translation Engine**: Handles multilingual translations

## 🎯 Use Cases

### 🏢 Business
- Summarize market research reports
- Translate business documents
- Analyze competitor content

### 🎓 Education
- Summarize research papers
- Translate academic content
- Study aid for language learning

### 📰 Content Creation
- Localize articles for different regions
- Create multilingual content
- Research and content curation

### 👥 Personal Use
- Quick understanding of foreign articles
- Language learning assistance
- Personal research organization

## 🔧 Configuration

### Processing Options
- **Summary Length**: Short, Medium, Long
- **Target Languages**: Multiple simultaneous translations
- **Source Language**: Auto-detection with manual override

## 📊 Performance

- **Summarization**: 85-95% content retention
- **Translation**: High accuracy across languages
- **Speed**: 2-10 seconds processing time
- **Web Extraction**: Robust with multiple fallback strategies

## 🤝 How to Contribute

We welcome contributions! Here's how you can help:

### 🐛 Report Bugs
- Use the Issues section
- Describe the problem clearly
- Include steps to reproduce

### 💡 Suggest Features
- Open an issue with your idea
- Explain how it would be useful

### 🔧 Code Contributions
1. Fork the project
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a Pull Request

### Areas Needing Help
- New language support
- UI/UX improvements
- Performance optimization
- Documentation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Getting Help

- **Documentation**: Check this README
- **Issues**: Open a GitHub issue
- **Questions**: Use the discussions section

## 🙏 Acknowledgments

- **Hugging Face** for transformer models
- **Streamlit** for the web framework
- **Facebook AI** for M2M100 and MBART models
- **Helsinki NLP** for MarianMT models

---

<div align="center">

**Made with ❤️ by Team SummarEase**

🤖 **Fatima Zahra** - Models & Deployment  
🎨 **Najlae** - Visual Interface  
📊 **Ikram** - Data Preparation

*Making multilingual content accessible to everyone*

</div>

## 📞 Contact

Have questions or suggestions? Feel free to reach out through GitHub issues or discussions!

---

**Happy Summarizing!** 📚✨