"""
YouTube Comment Sentiment Analyzer Dashboard
Streamlit web application for analyzing YouTube video comments
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fetch_comments import fetch_comments
from preprocess import preprocess_comments, truncate_long_comments
from sentiment_analysis import analyze_sentiment, get_sentiment_summary
from summarizer import generate_summary, get_top_comments_by_sentiment
from comment_responder import generate_comment_responses, get_response_summary


def main():
    st.set_page_config(
        page_title="YouTube Comment Sentiment Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 YouTube Comment Sentiment Analyzer")
    st.markdown("Analyze the sentiment of YouTube video comments with AI-powered insights")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys (Optional)")
        youtube_api_key = st.text_input(
            "YouTube Data API v3 Key",
            type="password",
            placeholder="Enter your YouTube API key (optional)",
            help="Get your API key from Google Cloud Console. If not provided, we'll use a fallback method."
        )
        
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password", 
            placeholder="Enter your Groq API key (optional)",
            help="Get your API key from Groq Console for AI-powered summaries."
        )
        
        # Processing options
        st.subheader("Processing Options")
        max_comments = st.slider("Maximum Comments to Analyze", 50, 500, 100)
        apply_lemmatization = st.checkbox("Apply Lemmatization", value=True)
        
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
            help="Enter the full YouTube video URL"
        )
    
    with col2:
        st.write("")  # Spacing
        analyze_button = st.button("🔍 Analyze Comments", type="primary")
    
    # Analysis section
    if analyze_button and video_url:
        if not video_url.strip():
            st.error("Please enter a valid YouTube video URL")
            return
        
        # Initialize session state for results
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Fetch comments
            status_text.text("Fetching comments...")
            progress_bar.progress(20)
            
            comments = fetch_comments(video_url, youtube_api_key, max_comments)
            
            if not comments:
                st.error("No comments found. Please check the video URL or try a different video.")
                return
            
            st.success(f"✅ Fetched {len(comments)} comments")
            
            # Step 2: Preprocess comments
            status_text.text("Preprocessing comments...")
            progress_bar.progress(40)
            
            processed_comments = preprocess_comments(comments, apply_lemmatization)
            processed_comments = truncate_long_comments(processed_comments)
            
            if not processed_comments:
                st.error("No valid comments after preprocessing.")
                return
            
            # Step 3: Analyze sentiment
            status_text.text("Analyzing sentiment...")
            progress_bar.progress(60)
            
            analyzed_comments = analyze_sentiment(processed_comments)
            
            # Step 4: Generate summary
            status_text.text("Generating summary...")
            progress_bar.progress(80)
            
            summary = generate_summary(analyzed_comments, groq_api_key)
            
            # Step 5: Generate responses to comments
            status_text.text("Generating responses to comments...")
            progress_bar.progress(90)
            
            comments_with_responses = generate_comment_responses(analyzed_comments, groq_api_key)
            
            # Step 6: Prepare results
            status_text.text("Preparing results...")
            progress_bar.progress(100)
            
            # Store results in session state
            st.session_state.analysis_results = {
                'comments': comments_with_responses,
                'summary': summary,
                'stats': get_sentiment_summary(comments_with_responses),
                'response_stats': get_response_summary(comments_with_responses)
            }
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            return
    
    # Display results
    if st.session_state.get('analysis_results'):
        results = st.session_state.analysis_results
        comments = results['comments']
        summary = results['summary']
        stats = results['stats']
        response_stats = results.get('response_stats', {})
        
        # Summary section
        st.header("📝 Summary of Main Points")
        st.info(summary)
        
        # Statistics section
        st.header("📈 Sentiment Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments", stats['total_comments'])
        
        with col2:
            st.metric(
                "Positive", 
                stats['positive_count'],
                f"{stats['positive_percentage']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Negative", 
                stats['negative_count'],
                f"{stats['negative_percentage']:.1f}%"
            )
        
        with col4:
            st.metric(
                "Responses Generated", 
                response_stats.get('total_responses', 0),
                f"{(response_stats.get('total_responses', 0) / stats['total_comments'] * 100):.1f}%" if stats['total_comments'] > 0 else "0%"
            )
        
        # Visualization section
        st.header("📊 Sentiment Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart with only positive and negative
            fig_pie = px.pie(
                values=[stats['positive_count'], stats['negative_count']],
                names=['Positive', 'Negative'],
                title="Sentiment Distribution",
                color_discrete_map={
                    'Positive': '#2E8B57',  # Green
                    'Negative': '#DC143C'   # Red
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart with only positive and negative
            fig_bar = px.bar(
                x=['Positive', 'Negative'],
                y=[stats['positive_count'], stats['negative_count']],
                title="Comment Count by Sentiment",
                color=['Positive', 'Negative'],
                color_discrete_map={
                    'Positive': '#2E8B57',  # Green
                    'Negative': '#DC143C'   # Red
                }
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Comments table
        st.header("💬 Comments with Sentiment Analysis & Responses")
        
        # Prepare data for display
        display_data = []
        for comment in comments:
            display_data.append({
                'Author': comment.get('author', 'Unknown'),
                'Comment': comment.get('original_text', '')[:150] + ('...' if len(comment.get('original_text', '')) > 150 else ''),
                'Sentiment': comment.get('sentiment', 'neutral').title(),
                'Confidence': f"{comment.get('confidence', 0):.2f}",
                'Your Response': comment.get('response', 'No response generated'),
                'Likes': comment.get('likes', 0)
            })
        
        df = pd.DataFrame(display_data)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ['All', 'Positive', 'Negative']  # Removed 'Neutral' option
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ['Confidence', 'Likes', 'Author']
            )
        
        # Apply filters
        if sentiment_filter != 'All':
            df = df[df['Sentiment'] == sentiment_filter]
        
        # Sort data
        if sort_by == 'Confidence':
            df = df.sort_values('Confidence', ascending=False)
        elif sort_by == 'Likes':
            df = df.sort_values('Likes', ascending=False)
        else:
            df = df.sort_values('Author')
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Comment": st.column_config.TextColumn(
                    "Comment",
                    width="medium",
                ),
                "Your Response": st.column_config.TextColumn(
                    "Your Response",
                    width="medium",
                ),
            }
        )
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="youtube_sentiment_analysis.csv",
            mime="text/csv"
        )
    
if __name__ == "__main__":
    main()
