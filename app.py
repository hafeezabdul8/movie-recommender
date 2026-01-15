import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import datetime

# ────────────────────────────────────────────────────────────────
# Modern dark theme + responsive styling
# ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .movie-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.5); }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; max-width: none !important; }
    img { border-radius: 8px; object-fit: cover; width: 100%; height: auto; }
    .stButton > button { background-color: #ff4b4b; color: white; border: none; border-radius: 8px; padding: 10px 20px; margin-top: 8px; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Feezman Movie Recommender ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto"
)

# ────────────────────────────────────────────────────────────────
# TMDB Config
# ────────────────────────────────────────────────────────────────
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
YOUTUBE_BASE = "https://www.youtube.com/watch?v="

# ────────────────────────────────────────────────────────────────
# Fetch trending movies (today's trending)
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)  # 30 min cache
def fetch_trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        movies = data.get('results', [])[:12]  # top 12 trending
        return pd.DataFrame([
            {
                'id': m['id'],
                'title': m['title'],
                'overview': m.get('overview', ''),
                'vote_average': m.get('vote_average', 0),
                'poster_path': m.get('poster_path'),
                'release_date': m.get('release_date', '')
            } for m in movies
        ])
    except:
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────
# Fetch new releases (recent + upcoming)
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)  # 1 hour cache
def fetch_new_movies():
    today = datetime.date.today()
    url = (
        f"https://api.themoviedb.org/3/discover/movie"
        f"?api_key={TMDB_API_KEY}"
        f"&language=en-US"
        f"&sort_by=release_date.desc"
        f"&include_adult=false"
        f"&include_video=false"
        f"&page=1"
        f"&primary_release_date.lte={today}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        movies = data.get('results', [])[:12]
        return pd.DataFrame([
            {
                'id': m['id'],
                'title': m['title'],
                'overview': m.get('overview', ''),
                'vote_average': m.get('vote_average', 0),
                'poster_path': m.get('poster_path'),
                'release_date': m.get('release_date', '')
            } for m in movies
        ])
    except:
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────
# Poster & Trailer
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_poster_url(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return TMDB_IMAGE_BASE + poster_path
    except:
        pass
    return None

@st.cache_data(ttl=86400)
def get_trailer_url(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5).json()
        for v in data.get('results', []):
            if v.get('site') == 'YouTube' and v.get('type') == 'Trailer':
                key = v.get('key')
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
        for v in data.get('results', []):
            if v.get('site') == 'YouTube':
                key = v.get('key')
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
    except:
        pass
    return None

# ────────────────────────────────────────────────────────────────
# Main UI with Trending & New tabs
# ────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 FEEZMAN MOVIE RECOMMENDER")
    st.markdown("Discover **trending** and **new** movies with posters & trailers!")

    tab1, tab2 = st.tabs(["Trending Now 🔥", "New Releases 🎥"])

    with tab1:
        st.subheader("Today's Trending Movies")
        trending_df = fetch_trending_movies()
        if trending_df.empty:
            st.warning("Could not load trending movies – try again later")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(trending_df.iterrows()):
                with cols[i % 4]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    poster = get_poster_url(row['id'])
                    st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"★ {row['vote_average']:.1f}")
                    trailer = get_trailer_url(row['id'])
                    if trailer:
                        st.markdown(f"[Watch Trailer]({trailer})", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Latest New Releases")
        new_df = fetch_new_movies()
        if new_df.empty:
            st.warning("Could not load new releases – try again later")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(new_df.iterrows()):
                with cols[i % 4]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    poster = get_poster_url(row['id'])
                    st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"★ {row['vote_average']:.1f} • {row['release_date']}")
                    trailer = get_trailer_url(row['id'])
                    if trailer:
                        st.markdown(f"[Watch Trailer]({trailer})", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Your existing recommendation section can stay below or be removed if you prefer only trending/new
    st.markdown("---")
    st.subheader("Personalized Recommendations")
    # ... add your original recommendation logic here if you still want it ...

if __name__ == "__main__":
    main()