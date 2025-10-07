
import os
import re
from typing import List, Dict, Optional
from youtube_comment_downloader import YoutubeCommentDownloader
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import streamlit as st


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_comments_api(video_id: str, api_key: str, max_results: int = 100) -> List[Dict]:
    """Fetch comments using YouTube Data API v3"""
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        comments = []
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=min(max_results, 100),
            order='relevance'
        )
        
        while request and len(comments) < max_results:
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textDisplay'],
                    'author': comment['authorDisplayName'],
                    'likes': comment['likeCount'],
                    'published': comment['publishedAt']
                })
            
            # Get next page if available
            if 'nextPageToken' in response and len(comments) < max_results:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=min(max_results - len(comments), 100),
                    pageToken=response['nextPageToken'],
                    order='relevance'
                )
            else:
                break
                
        return comments
        
    except HttpError as e:
        st.error(f"YouTube API Error: {e}")
        return []
    except Exception as e:
        st.error(f"Error fetching comments with API: {e}")
        return []


def fetch_comments_downloader(video_id: str, max_results: int = 100) -> List[Dict]:
    try:
        downloader = YoutubeCommentDownloader()
        comments = []
        
        for comment in downloader.get_comments_from_url(f'https://www.youtube.com/watch?v={video_id}'):
            if len(comments) >= max_results:
                break
                
            comments.append({
                'text': comment.get('text', ''),
                'author': comment.get('author', 'Unknown'),
                'likes': comment.get('votes', 0),
                'published': comment.get('time', '')
            })
            
        return comments
        
    except Exception as e:
        st.error(f"Error fetching comments with downloader: {e}")
        return []


def fetch_comments(video_url: str, api_key: Optional[str] = None, max_results: int = 100) -> List[Dict]:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please provide a valid YouTube video URL.")
        return []
    
    st.info(f"Extracting comments from video ID: {video_id}")
    
    # Try API first if key is provided
    if api_key and api_key.strip():
        st.info("Using YouTube Data API v3...")
        comments = fetch_comments_api(video_id, api_key.strip(), max_results)
        if comments:
            return comments
        else:
            st.warning("API failed, falling back to comment downloader...")
    
    # Fallback to downloader
    st.info("Using youtube-comment-downloader (no API key required)...")
    return fetch_comments_downloader(video_id, max_results)
