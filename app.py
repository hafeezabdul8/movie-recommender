import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

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
    page_title="Feezman Movie Recommender",
    layout="wide",
    initial_sidebar_state="auto"
)

# ────────────────────────────────────────────────────────────────
# Initialize session state
# ────────────────────────────────────────────────────────────────
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}  # movie_id: rating (1-5)

# ────────────────────────────────────────────────────────────────
# TMDB Config
# ────────────────────────────────────────────────────────────────
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
YOUTUBE_BASE = "https://www.youtube.com/watch?v="

# ────────────────────────────────────────────────────────────────
# Mood mapping (keep your existing)
# ────────────────────────────────────────────────────────────────
MOOD_TO_GENRES = {
    'happy': {'comedy': 1.8, 'animation': 1.5, 'adventure': 1.3, 'musical': 1.5},
    'sad': {'drama': 2.0, 'romance': 1.8, 'family': 1.5},
    'excited': {'action': 2.0, 'thriller': 1.8, 'sci-fi': 1.6, 'adventure': 1.5},
    'relaxed': {'comedy': 1.6, 'romance': 1.5, 'animation': 1.4, 'drama': 1.2},
    'scared': {'horror': 2.0, 'thriller': 1.8, 'mystery': 1.5},
    'romantic': {'romance': 2.0, 'drama': 1.6, 'comedy': 1.4},
    'angry': {'action': 1.8, 'thriller': 1.6, 'crime': 1.5},
    'thoughtful': {'drama': 1.8, 'biography': 1.6, 'sci-fi': 1.4}
}

def detect_mood_keywords(mood_text):
    if not mood_text:
        return {}
    mood_text = mood_text.lower()
    detected = {}
    for mood, boosts in MOOD_TO_GENRES.items():
        if any(word in mood_text for word in [mood] + list(boosts.keys())):
            detected[mood] = boosts
    return detected

# ────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    useful_cols = ['id', 'title', 'genres', 'overview', 'keywords', 'vote_average', 'vote_count']
    df = df[useful_cols]
    
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('[]')
    df['keywords'] = df['keywords'].fillna('[]')
    
    def extract_genre_names(json_str):
        try:
            genres_list = json.loads(json_str)
            return [genre['name'].lower() for genre in genres_list]
        except:
            return []
    
    df['genre_names'] = df['genres'].apply(extract_genre_names)
    
    def clean_text(text):
        return text.lower().replace(',', ' ').replace('"', '').replace("'", '')
    
    df['genres_clean'] = df['genre_names'].apply(lambda x: ' '.join(x))
    df['keywords'] = df['keywords'].apply(clean_text)
    df['overview'] = df['overview'].apply(clean_text)
    
    df['combined_features'] = df['genres_clean'] + ' ' + df['overview'] + ' ' + df['keywords']
    df = df[df['combined_features'].str.strip() != '']
    
    all_genres = sorted(set(genre for genres in df['genre_names'] for genre in genres))
    
    return df, all_genres

@st.cache_data
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, df.reset_index(drop=True)

# ────────────────────────────────────────────────────────────────
# Poster & Trailer
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_poster_url(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
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
        response = requests.get(url, timeout=5)
        data = response.json()
        results = data.get('results', [])
        for video in results:
            if video.get('site') == 'YouTube' and video.get('type') == 'Trailer':
                key = video.get('key')
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
        for video in results:
            if video.get('site') == 'YouTube':
                key = video.get('key')
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
    except:
        pass
    return None

# ────────────────────────────────────────────────────────────────
# Recommendation function
# ────────────────────────────────────────────────────────────────
def get_recommendations(title, df, cosine_sim, n=8, mood_boosts=None, selected_genres=None):
    title_clean = title.lower().strip()
    
    exact_match = df[df['title'].str.lower().str.strip() == title_clean]
    
    if not exact_match.empty:
        selected = exact_match.iloc[0]
        selected_title = selected['title']
        idx = exact_match.index[0]
    else:
        partial_matches = df[df['title'].str.lower().str.contains(title_clean, na=False)]
        if partial_matches.empty:
            return None, None, f"No movie found containing '{title}' 😢"
        partial_matches = partial_matches.sort_values('vote_count', ascending=False)
        selected = partial_matches.iloc[0]
        selected_title = selected['title']
        idx = selected.name
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    if mood_boosts:
        mood_boost_scores = np.zeros(len(df))
        for boosts in mood_boosts.values():
            for genre, weight in boosts.items():
                genre_mask = df['genres_clean'].str.contains(genre, case=False)
                mood_boost_scores[genre_mask] += weight
        if mood_boost_scores.max() > 0:
            mood_boost_scores /= mood_boost_scores.max()
        combined_scores = [0.7 * s + 0.3 * mood_boost_scores[i] for i, s in sim_scores]
        sim_scores = list(enumerate(combined_scores))
    
    if selected_genres:
        genre_mask = df['genres_clean'].apply(lambda x: any(g in x for g in selected_genres))
        sim_scores = [s for s in sim_scores if genre_mask[s[0]]]
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    
    if not sim_scores:
        return None, selected_title, "No recommendations match the selected genres 😔"
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices]
    
    return recommendations, selected_title, None

# ────────────────────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 FEEZMAN MOVIE RECOMMENDER")
    st.markdown("Rate, save to watchlist & discover movies!")
    
    df, all_genres = load_and_prepare_data()
    cosine_sim, df_indexed = build_similarity_matrix(df)
    
    # Sidebar: Watchlist & Ratings
    with st.sidebar:
        st.header("My Watchlist ❤️")
        if st.session_state.watchlist:
            watchlist_df = df[df['id'].isin(st.session_state.watchlist)]
            for _, movie in watchlist_df.iterrows():
                st.markdown(f"- **{movie['title']}** (★ {movie['vote_average']:.1f})")
                poster = get_poster_url(movie['id'])
                if poster:
                    st.image(poster, width=100)
            
            if st.button("🗑️ Clear Watchlist"):
                st.session_state.watchlist = []
                st.success("Watchlist cleared!")
                st.rerun()
        else:
            st.info("Your watchlist is empty – add some movies!")
        
        st.markdown("---")
        st.header("My Ratings")
        if st.session_state.user_ratings:
            for movie_id, rating in st.session_state.user_ratings.items():
                movie = df[df['id'] == movie_id]
                if not movie.empty:
                    title = movie['title'].iloc[0]
                    st.markdown(f"- **{title}**: {'★' * rating}")
        else:
            st.info("Rate some movies!")

    movie_list = sorted(df['title'].unique().tolist())
    movie_title = st.selectbox("Choose or type a movie:", options=movie_list)
    
    selected_genres = st.multiselect(
        "Filter by genres (optional)",
        options=all_genres,
        default=[],
        format_func=lambda x: x.title()
    )
    
    mood_input = st.text_input("How are you feeling? (optional)", "")
    
    n_recs = st.slider("Number of recommendations", 4, 12, 7)
    
    if st.button("Get Recommendations ✨"):
        mood_boosts = detect_mood_keywords(mood_input) if mood_input else None
        
        with st.spinner("Finding matches..."):
            recommendations, selected_title, error = get_recommendations(
                movie_title, df_indexed, cosine_sim, n_recs, mood_boosts, selected_genres
            )
            
            if error:
                st.error(error)
            else:
                notes = []
                if selected_genres:
                    notes.append(f"filtered to {', '.join(g.title() for g in selected_genres)}")
                if mood_boosts:
                    notes.append(f"mood: {', '.join(mood_boosts.keys())}")
                note_str = f" ({' + '.join(notes)})" if notes else ""
                st.subheader(f"Movies similar to: **{selected_title}**{note_str}")
                
                screen_width = st.session_state.get('screen_width', 1200)
                num_cols = 4 if screen_width >= 1024 else 3 if screen_width >= 768 else 2 if screen_width >= 480 else 1
                
                cols = st.columns(num_cols)
                
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    col = cols[i % num_cols]
                    with col:
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        
                        poster = get_poster_url(row['id'])
                        if poster:
                            st.image(poster, use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/300x450/1e1e2e/ffffff?text=No+Poster",
                                use_container_width=True
                            )
                            st.markdown(f"[View on TMDB](https://www.themoviedb.org/movie/{row['id']})", unsafe_allow_html=True)
                        
                        rating = f"★ {row['vote_average']:.1f}" if row['vote_average'] > 0 else "N/A"
                        st.markdown(f"**{row['title']}**  \n{rating}")
                        
                        trailer_url = get_trailer_url(row['id'])
                        if trailer_url:
                            st.markdown(f"[Watch Trailer 🎥]({trailer_url})", unsafe_allow_html=True)
                        else:
                            st.caption("No trailer available")
                        
                        # Watchlist button with rerun
                        movie_id = row['id']
                        movie_title = row['title']
                        
                        if movie_id not in st.session_state.watchlist:
                            if st.button("❤️ Add to Watchlist", key=f"add_{movie_id}_{i}"):
                                st.session_state.watchlist.append(movie_id)
                                st.success(f"Added **{movie_title}**!")
                                st.rerun()
                        else:
                            if st.button("💔 Remove", key=f"remove_{movie_id}_{i}"):
                                st.session_state.watchlist.remove(movie_id)
                                st.success(f"Removed **{movie_title}**!")
                                st.rerun()
                        
                        # User Rating (1-5 stars)
                        current_rating = st.session_state.user_ratings.get(movie_id, 0)
                        st.markdown("**Your rating:**")
                        cols_rating = st.columns(5)
                        for star in range(1, 6):
                            with cols_rating[star-1]:
                                if st.button(f"{'★' if star <= current_rating else '☆'}", key=f"rate_{movie_id}_{star}_{i}"):
                                    st.session_state.user_ratings[movie_id] = star
                                    st.success(f"Rated **{movie_title}** {star} stars!")
                                    st.rerun()
                        
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()