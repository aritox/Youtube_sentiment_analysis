"""
Comment Summarization Module
Generates summaries of main points from YouTube comments using Groq API or local models
"""

from typing import List, Dict, Optional
import streamlit as st
from groq import Groq
import os


def summarize_with_groq(comments: List[Dict], api_key: str) -> str:
    try:
        client = Groq(api_key=api_key)
        
        # Prepare comments text for summarization
        comments_text = []
        for comment in comments[:50]:  # Limit to first 50 comments to avoid token limits
            text = comment.get('original_text', comment.get('text', ''))
            if text and len(text.strip()) > 10:  # Only meaningful comments
                comments_text.append(f"- {text[:200]}")  # Truncate long comments
        
        if not comments_text:
            return "No meaningful comments found to summarize."
        
        # Create prompt for summarization
        prompt = f"""
        Please analyze the following YouTube comments and provide a concise summary of the main points, themes, and opinions expressed by viewers. Focus on the most common topics and sentiments.

        Comments:
        {chr(10).join(comments_text[:30])}  # Limit to avoid token limits

        Please provide a summary in 3-4 sentences covering:
        1. Main topics discussed
        2. Overall sentiment/tone
        3. Key concerns or praise mentioned
        4. Any notable patterns or trends

        Summary:
        """
        
        # Make API call
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            max_tokens=300,
            temperature=0.3
        )
        
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"Error with Groq API: {e}")
        return None


def summarize_locally(comments: List[Dict]) -> str:
    """
    Generate a simple local summary without external APIs
    """
    try:
        if not comments:
            return "No comments available for summarization."
        
        # Basic analysis
        total_comments = len(comments)
        
        # Count sentiments if available
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for comment in comments:
            sentiment = comment.get('sentiment', 'neutral')
            sentiment_counts[sentiment] += 1
        
        # Find most common words (simple approach)
        all_text = ' '.join([
            comment.get('processed_text', comment.get('original_text', ''))
            for comment in comments[:30]  # Limit for performance
        ])
        
        words = all_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        common_topics = [word for word, count in top_words if count > 2]
        
        # Generate summary
        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        sentiment_percentage = (sentiment_counts[dominant_sentiment] / total_comments) * 100
        
        summary = f"Analysis of {total_comments} comments shows a predominantly {dominant_sentiment} sentiment ({sentiment_percentage:.1f}%). "
        
        if common_topics:
            summary += f"Common topics discussed include: {', '.join(common_topics[:5])}. "
        
        summary += f"The comments contain {sentiment_counts['positive']} positive, {sentiment_counts['negative']} negative, and {sentiment_counts['neutral']} neutral responses."
        
        return summary
        
    except Exception as e:
        st.error(f"Error in local summarization: {e}")
        return "Unable to generate summary due to processing error."


def generate_summary(comments: List[Dict], groq_api_key: Optional[str] = None) -> str:
    """
    Main function to generate comment summary
    Uses Groq API if available, otherwise falls back to local summarization
    """
    if not comments:
        return "No comments available for summarization."
    
    # Try Groq API first if key is provided
    if groq_api_key and groq_api_key.strip():
        st.info("Generating summary using Groq API...")
        groq_summary = summarize_with_groq(comments, groq_api_key.strip())
        if groq_summary:
            return groq_summary
        else:
            st.warning("Groq API failed, using local summarization...")
    
    # Fallback to local summarization
    st.info("Generating summary using local analysis...")
    return summarize_locally(comments)


def get_top_comments_by_sentiment(comments: List[Dict], sentiment: str, limit: int = 5) -> List[Dict]:
    """
    Get top comments for a specific sentiment based on confidence and likes
    """
    filtered_comments = [
        comment for comment in comments 
        if comment.get('sentiment') == sentiment
    ]
    
    # Sort by confidence and likes
    sorted_comments = sorted(
        filtered_comments,
        key=lambda x: (x.get('confidence', 0) * 0.7 + (x.get('likes', 0) / 100) * 0.3),
        reverse=True
    )
    
    return sorted_comments[:limit]
