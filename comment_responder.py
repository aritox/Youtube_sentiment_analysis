"""
Comment Response Generator
Generates personalized responses to YouTube comments using AI
"""

import streamlit as st
from groq import Groq
import time
import random


def generate_comment_responses(comments, groq_api_key=None):
    
    if not comments:
        return []
    
    # Add responses to each comment
    for comment in comments:
        try:
            if groq_api_key:
                response = _generate_ai_response(comment, groq_api_key)
            else:
                response = _generate_fallback_response(comment)
            
            comment['response'] = response
            
        except Exception as e:
            st.warning(f"Error generating response for comment: {e}")
            comment['response'] = _generate_fallback_response(comment)
    
    return comments


def _generate_ai_response(comment, groq_api_key):
    try:
        client = Groq(api_key=groq_api_key)
        
        author = comment.get('author', 'there')
        original_text = comment.get('original_text', '')
        sentiment = comment.get('sentiment', 'neutral')
        
        # Create a prompt for response generation
        prompt = f"""
        You are a friendly content creator responding to YouTube comments. 
        Generate a brief, personalized response (max 20 words) to this comment:
        
        Author: {author}
        Comment: "{original_text}"
        Sentiment: {sentiment}
        
        Guidelines:
        - Be warm and engaging
        - Address the commenter by name when appropriate
        - Match the tone of the comment
        - Keep it conversational and authentic
        - For positive comments: show appreciation
        - For negative comments: be understanding and constructive
        
        Response:
        """
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.warning(f"AI response generation failed: {e}")
        return _generate_fallback_response(comment)


def _generate_fallback_response(comment):
    author = comment.get('author', 'there')
    sentiment = comment.get('sentiment', 'neutral')
    original_text = comment.get('original_text', '').lower()
    
    # Template responses based on sentiment and content
    if sentiment == 'positive':
        if any(word in original_text for word in ['thank', 'thanks', 'great', 'awesome', 'love', 'amazing']):
            responses = [
                f"Thank you so much {author}! 😊",
                f"Really appreciate it {author}! ❤️",
                f"Thanks {author}, that means a lot!",
                f"So glad you enjoyed it {author}! 🙏"
            ]
        elif any(word in original_text for word in ['good', 'nice', 'well done', 'excellent']):
            responses = [
                f"Thank you {author}! 🙌",
                f"Appreciate the kind words {author}!",
                f"Thanks for watching {author}! 😊"
            ]
        else:
            responses = [
                f"Thanks {author}! 😊",
                f"Appreciate you {author}! ❤️",
                f"Thank you for the support {author}!"
            ]
    
    elif sentiment == 'negative':
        if any(word in original_text for word in ['bad', 'terrible', 'hate', 'worst']):
            responses = [
                f"Sorry to hear that {author}. I'll work on improving! 🙏",
                f"Thanks for the feedback {author}, I appreciate your honesty.",
                f"I understand {author}, I'll keep working to do better!"
            ]
        elif any(word in original_text for word in ['confusing', 'unclear', 'hard']):
            responses = [
                f"Thanks for pointing that out {author}! I'll try to explain better next time.",
                f"Good feedback {author}, I'll work on making it clearer!",
                f"Appreciate the input {author}, clarity is important!"
            ]
        else:
            responses = [
                f"Thanks for the feedback {author}! 🙏",
                f"I appreciate your perspective {author}.",
                f"Thank you for sharing your thoughts {author}!"
            ]
    
    else:  # neutral
        responses = [
            f"Thanks for watching {author}! 😊",
            f"Appreciate the comment {author}!",
            f"Thanks for being here {author}! 🙏",
            f"Great to see you {author}!"
        ]
    
    return random.choice(responses)


@st.cache_data
def get_response_summary(comments):
    if not comments:
        return {}
    
    total_responses = len([c for c in comments if c.get('response')])
    ai_responses = len([c for c in comments if c.get('response') and len(c.get('response', '')) > 30])
    template_responses = total_responses - ai_responses
    
    return {
        'total_responses': total_responses,
        'ai_responses': ai_responses,
        'template_responses': template_responses
    }
