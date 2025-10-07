"""
Text Preprocessing Module
Cleans and preprocesses YouTube comments for sentiment analysis
"""

import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        st.warning(f"Could not download NLTK data: {e}")
        return False


def remove_urls(text: str) -> str:
    """Remove URLs from text"""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text"""
    html_pattern = re.compile(r'<.*?>')
    return html_pattern.sub('', text)


def remove_emojis(text: str) -> str:
    """Remove emojis from text"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_special_characters(text: str) -> str:
    """Remove special characters and extra whitespace"""
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords_text(text: str, languages: List[str] = ['english', 'french']) -> str:
    """Remove stopwords from text"""
    try:
        stop_words = set()
        for lang in languages:
            try:
                stop_words.update(stopwords.words(lang))
            except:
                continue
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    except:
        return text


def lemmatize_text(text: str) -> str:
    """Apply lemmatization to text"""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return ' '.join(lemmatized_words)
    except:
        return text.lower()


def preprocess_comment(text: str, apply_lemmatization: bool = True) -> str:
    """
    Complete preprocessing pipeline for a single comment
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Step 1: Remove URLs
    text = remove_urls(text)
    
    # Step 2: Remove HTML tags
    text = remove_html_tags(text)
    
    # Step 3: Remove emojis
    text = remove_emojis(text)
    
    # Step 4: Remove special characters
    text = remove_special_characters(text)
    
    # Step 5: Convert to lowercase
    text = text.lower()
    
    # Step 6: Remove stopwords
    text = remove_stopwords_text(text)
    
    # Step 7: Apply lemmatization (optional)
    if apply_lemmatization:
        text = lemmatize_text(text)
    
    return text.strip()


def preprocess_comments(comments: List[dict], apply_lemmatization: bool = True) -> List[dict]:
    """
    Preprocess a list of comments
    """
    # Ensure NLTK data is downloaded
    download_nltk_data()
    
    processed_comments = []
    
    for comment in comments:
        processed_comment = comment.copy()
        original_text = comment.get('text', '')
        
        # Preprocess the text
        processed_text = preprocess_comment(original_text, apply_lemmatization)
        
        # Keep both original and processed text
        processed_comment['original_text'] = original_text
        processed_comment['processed_text'] = processed_text
        
        # Only keep comments with meaningful content after preprocessing
        if len(processed_text.strip()) > 3:  # At least 3 characters
            processed_comments.append(processed_comment)
    
    return processed_comments


def truncate_long_comments(comments: List[dict], max_length: int = 200) -> List[dict]:
    """
    Truncate long comments to specified length for better processing
    """
    truncated_comments = []
    
    for comment in comments:
        truncated_comment = comment.copy()
        
        # Truncate original text
        if len(comment.get('original_text', '')) > max_length:
            truncated_comment['original_text'] = comment['original_text'][:max_length] + "..."
        
        # Truncate processed text
        if len(comment.get('processed_text', '')) > max_length:
            truncated_comment['processed_text'] = comment['processed_text'][:max_length] + "..."
        
        truncated_comments.append(truncated_comment)
    
    return truncated_comments
