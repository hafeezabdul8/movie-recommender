import streamlit as st
import pandas as pd
import requests
from datetime import datetime

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
    .stButton > button { background-color: #ff4b4b; color: white; border: none; border-radius: 8px; padding: 10px 20px; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Feezman Movies", page_icon="🎬", layout="wide")

# ────────────────────────────────────────────────────────────────
# TMDB Config – your key from secrets
# ────────────────────────────────────────────────────────────────
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ────────────────────────────────────────────────────────────────
# Fetch Trending Movies (today)
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)  # 30 min cache
def fetch_trending():
    url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url, timeout=8).json().get('results', [])[:12]
        return pd.DataFrame([{
            'id': m['id'],
            'title': m['title'],
            'vote_average': m.get('vote_average', 0),
            'poster_path': m.get('poster_path'),
            'release_date': m.get('release_date', '')
        } for m in data])
    except:
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────
# Fetch New Releases (latest by date)
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)  # 1 hour cache
def fetch_new():
    today = datetime.now().strftime("%Y-%m-%d")
    url = (
        f"https://api.themoviedb.org/3/discover/movie"
        f"?api_key={TMDB_API_KEY}&language=en-US"
        f"&sort_by=release_date.desc&include_adult=false"
        f"&page=1&primary_release_date.lte={today}"
    )
    try:
        data = requests.get(url, timeout=8).json().get('results', [])[:12]
        return pd.DataFrame([{
            'id': m['id'],
            'title': m['title'],
            'vote_average': m.get('vote_average', 0),
            'poster_path': m.get('poster_path'),
            'release_date': m.get('release_date', '')
        } for m in data])
    except:
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────
# Poster & Trailer helpers
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        poster = requests.get(url, timeout=5).json().get('poster_path')
        return TMDB_IMAGE_BASE + poster if poster else None
    except:
        return None

@st.cache_data(ttl=86400)
def get_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        videos = requests.get(url, timeout=5).json().get('results', [])
        for v in videos:
            if v.get('site') == 'YouTube' and v.get('type') == 'Trailer':
                return f"https://www.youtube.com/watch?v={v['key']}"
        return None
    except:
        return None

def fetch_current_movies():
    raise NotImplementedError

# ────────────────────────────────────────────────────────────────
# Main UI with Trending, New, and Search
# ────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 FEEZMAN MOVIES")
    st.markdown("Browse **trending** & **new** movies, or search by title/genre!")

    # Tabs for Trending, New, and Search
    tab_trend, tab_new, tab_search = st.tabs(["Trending 🔥", "New Releases 🎥", "Search"])

    # Trending Tab
    with tab_trend:
        st.subheader("Trending Movies Today (Worldwide)")
        df_trend = fetch_trending()
        if df_trend.empty:
            st.warning("Couldn't load trending – check your TMDB API key")
        else:
            cols = st.columns(4)
            for i, row in df_trend.iterrows():
                with cols[i % 4]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    poster = get_poster(row['id'])
                    st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"★ {row['vote_average']:.1f}")
                    trailer = get_trailer(row['id'])
                    if trailer:
                        st.markdown(f"[Trailer]({trailer})", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # New Releases Tab
    with tab_new:
        st.subheader("Latest New Releases")
        df_new = fetch_new()
        if df_new.empty:
            st.warning("Couldn't load new releases – check your TMDB API key")
        else:
            cols = st.columns(4)
            for i, row in df_new.iterrows():
                with cols[i % 4]:
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    poster = get_poster(row['id'])
                    st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"★ {row['vote_average']:.1f} • {row['release_date'][:4]}")
                    trailer = get_trailer(row['id'])
                    if trailer:
                        st.markdown(f"[Trailer]({trailer})", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    # Search Tab
    with tab_search:
        st.subheader("Search by Title or Genre")
        search_query = st.text_input("Enter movie title or genre(s)", placeholder="e.g. Inception, horror, comedy action")

        if st.button("Search"):
            if not search_query.strip():
                st.warning("Please enter a movie or genre")
            else:
                query_lower = search_query.lower().strip()

                # Load full recent movies for search
                df, all_genres = fetch_current_movies()  # Reuse function from earlier versions

                # Detect genre search
                matched_genres = [g for g in all_genres if g in query_lower]
                if matched_genres:
                    st.info(f"Showing movies matching genres: {', '.join(g.title() for g in matched_genres)}")
                    mask = df['genres_clean'].apply(lambda x: any(g in x for g in matched_genres))
                    results = df[mask].sort_values('vote_count', ascending=False).head(12)
                else:
                    # Title search
                    st.info(f"Searching for movies like: **{search_query}**")
                    results = df[df['title'].str.lower().str.contains(query_lower, na=False)]
                    if results.empty:
                        st.info("No exact match – showing closest titles")
                        results = df[df['title'].str.lower().str.contains(query_lower[:6], na=False)].head(12)

                if results.empty:
                    st.error("No results found")
                else:
                    cols = st.columns(4)
                    for i, row in results.iterrows():
                        with cols[i % 4]:
                            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                            poster = get_poster(row['id'])
                            st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                            st.markdown(f"**{row['title']}**")
                            st.caption(f"★ {row['vote_average']:.1f}")
                            trailer = get_trailer(row['id'])
                            if trailer:
                                st.markdown(f"[Trailer]({trailer})", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()