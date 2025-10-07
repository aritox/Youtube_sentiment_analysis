"""
Sentiment Analysis Module
Performs sentiment classification on preprocessed comments using Hugging Face transformers
"""

from typing import List, Dict
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    max_attempts = 3
    
    for attempt in range(1, max_attempts + 1):
        try:
            st.write(f"Loading sentiment model (Attempt {attempt}/{max_attempts})...")
            
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                return_all_scores=False  # Only return the top prediction
            )
            
            st.success("✅ Sentiment model loaded successfully!")
            return sentiment_pipeline
            
        except Exception as e:
            st.error(f"Error loading sentiment model: {str(e)}")
            
            if attempt < max_attempts:
                retry_delay = 2 ** (attempt - 1)  # Exponential backoff: 1, 2 seconds
                st.write(f"Retrying in {retry_delay} seconds... (Attempt {attempt}/{max_attempts})") 
                time.sleep(retry_delay)
            else:
                st.error("Could not load sentiment analysis model")
                return None
    
    return None


def map_sentiment_label(label: str, score: float) -> Dict[str, float]:
    
    if label == "POSITIVE":
        return {
            "positive": score,
            "negative": 1 - score
        }
    elif label == "NEGATIVE":
        return {
            "positive": 1 - score,
            "negative": score
        }
    else:
        # Default case - treat as neutral leaning positive
        return {
            "positive": 0.6,
            "negative": 0.4
        }


def classify_sentiment_batch(texts: List[str], sentiment_pipeline) -> List[Dict]:
    """
    Classify sentiment for a batch of texts
    """
    if not sentiment_pipeline:
        return []
    
    results = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Get predictions for batch
            batch_predictions = sentiment_pipeline(batch_texts)
            
            for prediction in batch_predictions:
                # Map to standardized format
                sentiment_scores = map_sentiment_label(
                    prediction['label'], 
                    prediction['score']
                )
                
                final_sentiment = "positive" if sentiment_scores['positive'] > sentiment_scores['negative'] else "negative"
                
                results.append({
                    'sentiment': final_sentiment,
                    'confidence': prediction['score'],
                    'scores': sentiment_scores
                })
                
        except Exception as e:
            st.warning(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add default results for failed batch
            for _ in batch_texts:
                results.append({
                    'sentiment': 'positive',  # Default to positive instead of neutral
                    'confidence': 0.5,
                    'scores': {'positive': 0.5, 'negative': 0.5}
                })
    
    return results


def analyze_sentiment(comments: List[Dict]) -> List[Dict]:
    """
    Main function to analyze sentiment of preprocessed comments
    """
    if not comments:
        return []
    
    # Load sentiment model
    sentiment_pipeline = load_sentiment_model()
    if not sentiment_pipeline:
        st.error("Could not load sentiment analysis model")
        return comments
    
    # Extract processed texts for analysis
    texts_to_analyze = []
    for comment in comments:
        text = comment.get('processed_text', comment.get('original_text', ''))
        # Ensure text is not empty and not too long
        if text and len(text.strip()) > 0:
            # Truncate if too long (model limit)
            if len(text) > 512:
                text = text[:512]
            texts_to_analyze.append(text)
        else:
            texts_to_analyze.append("okay comment")  # Neutral fallback
    
    # Perform sentiment analysis
    with st.spinner("Analyzing sentiment..."):
        sentiment_results = classify_sentiment_batch(texts_to_analyze, sentiment_pipeline)
    
    # Combine results with original comments
    analyzed_comments = []
    for i, comment in enumerate(comments):
        analyzed_comment = comment.copy()
        
        if i < len(sentiment_results):
            analyzed_comment.update(sentiment_results[i])
        else:
            # Fallback for missing results
            analyzed_comment.update({
                'sentiment': 'positive',  # Default to positive
                'confidence': 0.5,
                'scores': {'positive': 0.5, 'negative': 0.5}
            })
        
        analyzed_comments.append(analyzed_comment)
    
    return analyzed_comments


def get_sentiment_summary(analyzed_comments: List[Dict]) -> Dict:
    """
    Generate summary statistics for sentiment analysis results
    """
    if not analyzed_comments:
        return {
            'total_comments': 0,
            'positive_count': 0,
            'negative_count': 0,
            'positive_percentage': 0.0,
            'negative_percentage': 0.0
        }
    
    total_comments = len(analyzed_comments)
    positive_count = sum(1 for comment in analyzed_comments if comment.get('sentiment') == 'positive')
    negative_count = sum(1 for comment in analyzed_comments if comment.get('sentiment') == 'negative')
    
    return {
        'total_comments': total_comments,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': (positive_count / total_comments) * 100 if total_comments > 0 else 0.0,
        'negative_percentage': (negative_count / total_comments) * 100 if total_comments > 0 else 0.0
    }
