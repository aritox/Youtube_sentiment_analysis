# YouTube Comment Sentiment Analyzer Dashboard

A Streamlit web application that analyzes the sentiment of YouTube video comments using AI-powered natural language processing. It extracts comments, preprocesses text, performs sentiment classification, and provides an interactive dashboard with summaries and export options.

## Features

- **Comment Extraction**: Fetch comments via YouTube Data API v3 (primary) or a fallback downloader.
- **Text Preprocessing**: Remove URLs, emojis, HTML, special characters; lowercase; remove stopwords; optional lemmatization.
- **Sentiment Analysis**: Classify comments as positive, negative, or neutral (DistilBERT model).
- **AI Summarization**: Generate summaries (primary: Groq API with LLaMA 3.1 8B; fallback: local summarization).
- **Interactive Dashboard**: Streamlit UI with charts, tables and filters.
- **Export**: Download results as CSV.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/aritox/Youtube_sentiment_analysis.git
cd Youtube_sentiment_analysis
