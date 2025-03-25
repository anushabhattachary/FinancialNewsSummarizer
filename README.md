# FinancialNewsSummarizer
This financial news summarizer is a comprehensive tool designed to scrape, process, and summarize financial news from multiple sources, offering concise and insightful summaries using AI-powered NLP models.

# AI-Powered Financial News Summarizer

## Project Overview
The **AI-Powered Financial News Summarizer** is a comprehensive tool designed to scrape, process, and summarize financial news from multiple sources, offering concise and insightful summaries using AI-powered NLP models.

## Features
- **Multi-source financial news aggregation** (Yahoo Finance, MarketWatch, CNBC)
- **AI-powered summarization** using transformer-based models (T5, BART)
- **Web interface** for real-time summarization
- **Fine-tuned models** for financial domain-specific terminology
- **Batch processing** for efficient summarization

## Core Components
### News Scraper (scraper.py)
- Scrapes articles from multiple financial sources
- Extracts details: **title, content, URL, date**
- Implements **error handling** and **source-specific parsing**

### News Summarizer (summarizer.py)
- Uses **pre-trained NLP models (T5, BART)** for text summarization
- Features:
  - **Preprocessing of financial text**
  - **Configurable summary length**
  - **Financial term expansion**
  - **Batch summarization**
  - **Fine-tuning for domain-specific accuracy**

### Web Interface (interface.html)
- **Flask-based** web application
- Features:
  - **Real-time news aggregation**
  - **Custom text summarization**
  - **Interactive UI with responsive design**

## 📂 Key Python Scripts (run in terminal, in this order)
| File |
|------|
| `scraper.py` | 
| `summarizer.py` | 
| `application.py` |
|`fine_tuning.py`|
| `interface.html` | 

## Web Frontend
- Built with **HTML, CSS, Bootstrap**
- Responsive two-column layout
- Key Features:
  - **Live news display**
  - **Custom summarization input**
  - **User-friendly UI with loading states**

## Tech Stack
- **Python** (Flask, BeautifulSoup, Pandas, Requests)
- **NLP Libraries** (Hugging Face Transformers, PyTorch)
- **Web Technologies** (HTML5, CSS, Bootstrap, JavaScript)

## Setup & Installation
```bash
# Create virtual environment
python -m venv financial_summarizer_env

# Install dependencies
pip install transformers datasets pandas numpy requests beautifulsoup4 torch flask
```


## Future Enhancements
Expand to **more news sources**
Implement **advanced caching** for faster performance
Enhance **fine-tuning techniques** for better accuracy
Add **user authentication** and personalized news feeds
Incorporate **sentiment analysis** for market insights

## Challenges Solved
- Overcoming **web scraping restrictions** and **data inconsistencies**
- Improving **summarization quality** and **relevance** for financial news
- Handling **financial jargon and domain-specific terminology**

## AI & NLP Techniques Used
- **Web Scraping & Text Processing**
- **Transformer-based Text Summarization**
- **Model Fine-Tuning for Finance Domain**
- **REST API Development with Flask**
- **Frontend & Backend Integration**

---

Conclusion
This project demonstrates a **real-world AI application in finance**, integrating **NLP, web scraping, and web development** to provide an **automated, AI-powered news summarization tool**.

