import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
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
    page_title="Feezman Movie Recommender – New & Trending",
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
# Fetch current/recent movies from TMDB (live)
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)  # refresh every hour
def fetch_current_movies():
    today = datetime.date.today()
    current_year = today.year
    last_year = current_year - 1

    # Popular movies from last 2 years + this year
    url_popular = (
        f"https://api.themoviedb.org/3/discover/movie"
        f"?api_key={TMDB_API_KEY}"
        f"&language=en-US"
        f"&sort_by=popularity.desc"
        f"&include_adult=false"
        f"&include_video=false"
        f"&page=1"
        f"&primary_release_date.gte={last_year}-01-01"
        f"&primary_release_date.lte={today}"
    )

    # Newest releases (sorted by date)
    url_new = (
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
        popular_res = requests.get(url_popular, timeout=10).json()
        new_res = requests.get(url_new, timeout=10).json()

        movies = popular_res.get('results', [])[:30] + new_res.get('results', [])[:30]

        data = []
        for m in movies:
            data.append({
                'id': m['id'],
                'title': m['title'],
                'overview': m.get('overview', ''),
                'vote_average': m.get('vote_average', 0),
                'vote_count': m.get('vote_count', 0),
                'release_date': m.get('release_date', ''),
                'poster_path': m.get('poster_path'),
                'genre_ids': m.get('genre_ids', [])
            })

        df = pd.DataFrame(data).drop_duplicates(subset='id')

        # Get genre mapping from TMDB (only once)
        genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
        genre_res = requests.get(genre_url, timeout=5).json()
        genre_map = {g['id']: g['name'].lower() for g in genre_res.get('genres', [])}

        # Convert genre_ids to names
        df['genre_names'] = df['genre_ids'].apply(
            lambda ids: [genre_map.get(i, '') for i in ids if i in genre_map]
        )
        df['genres_clean'] = df['genre_names'].apply(lambda x: ' '.join(x))
        df['combined_features'] = df['genres_clean'] + ' ' + df['overview']

        # Remove rows with no useful features
        df = df[df['combined_features'].str.strip() != '']

        all_genres = sorted(set(g for genres in df['genre_names'] for g in genres))

        return df, all_genres

    except Exception as e:
        st.error(f"Could not fetch movies from TMDB: {e}")
        return pd.DataFrame(), []

# ────────────────────────────────────────────────────────────────
# Build similarity
# ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800)
def build_similarity_matrix(df):
    if df.empty:
        return None, df
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
        # fallback any youtube
        for v in data.get('results', []):
            if v.get('site') == 'YouTube':
                key = v.get('key')
                if key:
                    return f"https://www.youtube.com/watch?v={key}"
    except:
        pass
    return None

# ────────────────────────────────────────────────────────────────
# Recommendation function
# ────────────────────────────────────────────────────────────────
def get_recommendations(title, df, cosine_sim, n=8, mood_boosts=None, selected_genres=None):
    if cosine_sim is None:
        return None, None, "No data available"

    title_clean = title.lower().strip()
    
    exact = df[df['title'].str.lower().str.strip() == title_clean]
    
    if not exact.empty:
        selected = exact.iloc[0]
        selected_title = selected['title']
        idx = exact.index[0]
    else:
        partial = df[df['title'].str.lower().str.contains(title_clean, na=False)]
        if partial.empty:
            return None, None, f"No movie found for '{title}'"
        partial = partial.sort_values('vote_count', ascending=False)
        selected = partial.iloc[0]
        selected_title = selected['title']
        idx = selected.name
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    if mood_boosts:
        mood_boost_scores = np.zeros(len(df))
        for boosts in mood_boosts.values():
            for genre, w in boosts.items():
                mask = df['genres_clean'].str.contains(genre, case=False)
                mood_boost_scores[mask] += w
        if mood_boost_scores.max() > 0:
            mood_boost_scores /= mood_boost_scores.max()
        combined = [0.7 * s + 0.3 * b for s, b in zip(sim_scores, mood_boost_scores)]
        sim_scores = list(enumerate(combined))
    
    if selected_genres:
        mask = df['genres_clean'].apply(lambda x: any(g in x for g in selected_genres))
        sim_scores = [s for s in sim_scores if mask[s[0]]]
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]
    
    if not sim_scores:
        return None, selected_title, "No matching recommendations found"
    
    indices = [i[0] for i in sim_scores]
    return df.iloc[indices], selected_title, None

# ────────────────────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 FEEZMAN MOVIE RECOMMENDER")
    st.markdown("New & trending movies – with posters, trailers, mood & genre filter!")

    df, all_genres = fetch_current_movies()

    if df.empty:
        st.error("Could not load movies. Please try again later.")
        return

    cosine_sim, df_indexed = build_similarity_matrix(df)

    # New & Trending section
    st.subheader("New & Trending Right Now")
    trending = df.sort_values('vote_count', ascending=False).head(8)
    cols_trend = st.columns(4)
    for i, (_, row) in enumerate(trending.iterrows()):
        with cols_trend[i % 4]:
            poster = get_poster_url(row['id'])
            st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
            st.markdown(f"**{row['title']}**  \n★ {row['vote_average']:.1f}")

    movie_list = sorted(df['title'].unique().tolist())
    movie_title = st.selectbox("Choose or type a movie:", options=movie_list)

    selected_genres = st.multiselect(
        "Filter by genres",
        options=all_genres,
        default=[],
        format_func=lambda x: x.title()
    )

    mood_input = st.text_input("How are you feeling? (optional)", "")

    n_recs = st.slider("Number of recommendations", 4, 12, 8)

    if st.button("Get Recommendations ✨"):
        mood_boosts = detect_mood_keywords(mood_input)

        with st.spinner("Finding matches..."):
            recs, selected_title, error = get_recommendations(
                movie_title, df_indexed, cosine_sim, n_recs, mood_boosts, selected_genres
            )

            if error:
                st.error(error)
            else:
                note = []
                if selected_genres:
                    note.append(f"filtered to {', '.join(g.title() for g in selected_genres)}")
                if mood_boosts:
                    note.append(f"mood: {', '.join(mood_boosts)}")
                note_str = f" ({' + '.join(note)})" if note else ""
                st.subheader(f"Movies similar to **{selected_title}**{note_str}")

                num_cols = 4
                screen = st.session_state.get('screen_width', 1200)
                if screen < 480: num_cols = 1
                elif screen < 768: num_cols = 2
                elif screen < 1024: num_cols = 3

                cols = st.columns(num_cols)

                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % num_cols]:
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)

                        poster = get_poster_url(row['id'])
                        st.image(poster or "https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)

                        st.markdown(f"**{row['title']}**  \n★ {row['vote_average']:.1f}")

                        trailer = get_trailer_url(row['id'])
                        if trailer:
                            st.markdown(f"[🎥 Trailer]({trailer})", unsafe_allow_html=True)
                        else:
                            st.caption("No trailer")

                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()