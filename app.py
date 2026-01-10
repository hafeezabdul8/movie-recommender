import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ────────────────────────────────────────────────────────────────
# Modern styling + responsive tweaks (same as before)
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

st.set_page_config(page_title="Feeezman Movie Recommender", layout="wide", initial_sidebar_state="auto")

# ────────────────────────────────────────────────────────────────
# TMDB API Config
# ────────────────────────────────────────────────────────────────
TMDB_API_KEY = '7e4c7f413d3fbee94f9d6106052b7273'  # ← Replace with your real key!
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ────────────────────────────────────────────────────────────────
# Mood → Genre Boost Mapping (adjust weights as you like)
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
    mood_text = mood_text.lower()
    detected = {}
    for mood, boosts in MOOD_TO_GENRES.items():
        if any(word in mood_text for word in [mood] + [k for k in boosts]):
            detected[mood] = boosts
    return detected

# ────────────────────────────────────────────────────────────────
# Load data & build similarity (same as before)
# ────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('tmdb_5000_movies.csv')
    useful_cols = ['id', 'title', 'genres', 'overview', 'keywords', 'vote_average', 'vote_count']
    df = df[useful_cols]
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('[]')
    df['keywords'] = df['keywords'].fillna('[]')
    
    def clean_text(text):
        return text.lower().replace(',', ' ').replace('"', '').replace("'", '')
    
    df['genres'] = df['genres'].apply(clean_text)
    df['keywords'] = df['keywords'].apply(clean_text)
    df['overview'] = df['overview'].apply(clean_text)
    
    df['combined_features'] = df['genres'] + ' ' + df['overview'] + ' ' + df['keywords']
    df = df[df['combined_features'].str.strip() != '']
    return df

@st.cache_data
def build_similarity_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, df.reset_index(drop=True)

@st.cache_data(ttl=3600)
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

# ────────────────────────────────────────────────────────────────
# Recommendation function with mood boost
# ────────────────────────────────────────────────────────────────
def get_recommendations(title, df, cosine_sim, n=8, mood_boosts=None):
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
    
    # Base similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Apply mood boost if any
    if mood_boosts:
        mood_boost_scores = np.zeros(len(df))
        for boosts in mood_boosts.values():
            for genre, weight in boosts.items():
                genre_mask = df['genres'].str.contains(genre, case=False)
                mood_boost_scores[genre_mask] += weight
        # Normalize boost
        if mood_boost_scores.max() > 0:
            mood_boost_scores /= mood_boost_scores.max()
        # Combine: 0.7 similarity + 0.3 mood boost (adjustable)
        combined_scores = [0.7 * s + 0.3 * mood_boost_scores[i] for i, s in sim_scores]
        sim_scores = list(enumerate(combined_scores))
    
    # Sort & take top N (skip self)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices]
    
    return recommendations, selected_title, None

# ────────────────────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 FEEEZMAN MOVIE RECOMMENDER")
    st.markdown("Personalized movies with real posters – now with mood magic!")
    st.set_page_config(
    page_title="Feeezman Movies",
    page_icon="🎬",
    layout="wide"
)
    
    df = load_and_prepare_data()
    cosine_sim, df_indexed = build_similarity_matrix(df)
    
    movie_list = sorted(df['title'].unique().tolist())
    movie_title = st.selectbox("Choose or type a movie you love:", options=movie_list)
    
    mood_input = st.text_input("How are you feeling? (e.g. sad, excited, romantic, chill...)", "")
    
    n_recs = st.slider("Number of recommendations", 4, 12, 7)
    
    if st.button("Get Recommendations ✨"):
        mood_boosts = detect_mood_keywords(mood_input) if mood_input else None
        
        with st.spinner("Matching your mood & finding movies..."):
            recommendations, selected_title, error = get_recommendations(
                movie_title, df_indexed, cosine_sim, n_recs, mood_boosts
            )
            
            if error:
                st.error(error)
            else:
                mood_note = f" (Mood boost: {', '.join(mood_boosts.keys())})" if mood_boosts else ""
                st.subheader(f"Movies similar to: **{selected_title}**{mood_note}")
                
                # Adaptive columns (from previous responsive version)
                screen_width = st.session_state.get('screen_width', 1200)
                num_cols = 4
                if screen_width < 480: num_cols = 1
                elif screen_width < 768: num_cols = 2
                elif screen_width < 1024: num_cols = 3
                
                cols = st.columns(num_cols)
                
                for i, (_, row) in enumerate(recommendations.iterrows()):
                    col = cols[i % num_cols]
                    with col:
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        poster = get_poster_url(row['id'])
                        st.image(poster if poster else "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                        rating = f"★ {row['vote_average']:.1f}" if row['vote_average'] > 0 else "N/A"
                        st.markdown(f"**{row['title']}**  \n{rating}")
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()