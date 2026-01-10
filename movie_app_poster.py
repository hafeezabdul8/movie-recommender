import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import requests

# ===============================================
# TMDB API Config - PUT YOUR REAL KEY HERE!
# ===============================================
TMDB_API_KEY = '7e4c7f413d3fbee94f9d6106052b7273'  # ← Replace with your actual key!
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ===============================================
# Load & Train Model (cached)
# ===============================================
@st.cache_resource
def load_and_train():
    ratings = pd.read_csv('ml-100k/ua.base', 
                          sep='\t', 
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_factorized = svd.fit_transform(user_item_matrix)
    predicted_ratings = np.dot(matrix_factorized, svd.components_)
    
    movies = pd.read_csv('ml-100k/u.item',
                         sep='|',
                         encoding='latin-1',
                         header=None,
                         usecols=[0, 1],
                         names=['item_id', 'title'])
    
    # Extract year for better TMDB matching
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
    
    return user_item_matrix, predicted_ratings, movies

user_item_matrix, predicted_ratings, movies = load_and_train()

# ===============================================
# Recommendation function
# ===============================================
def get_recommendations(movie_title, n=8):
    # Fuzzy match on clean_title or full title
    match = movies[
        movies['clean_title'].str.contains(movie_title, case=False, na=False) |
        movies['title'].str.contains(movie_title, case=False, na=False)
    ]
    
    if match.empty:
        return None, f"No movie found with '{movie_title}' 😢 Try classics like 'Star Wars', 'Toy Story', 'Fargo'"
    
    input_movie = match.iloc[0]
    movie_id = input_movie['item_id']
    
    users_who_rated = user_item_matrix[movie_id][user_item_matrix[movie_id] > 3].index
    
    if len(users_who_rated) == 0:
        return None, "Not enough high ratings for this movie 😔"
    
    similar_user_preds = predicted_ratings[users_who_rated - 1]
    avg_preds = np.mean(similar_user_preds, axis=0)
    
    preds_series = pd.Series(avg_preds, index=user_item_matrix.columns)
    preds_series[movie_id] = -np.inf  # Exclude itself
    
    top_n_ids = preds_series.sort_values(ascending=False).head(n).index.tolist()
    
    top_movies = movies[movies['item_id'].isin(top_n_ids)]
    return top_movies, None

# ===============================================
# Fetch Poster from TMDB (improved: try with year)
# ===============================================
def get_poster_url(title, year=None):
    try:
        query = f"{title} {year}" if year else title
        params = {
            'api_key': TMDB_API_KEY,
            'query': query,
            'language': 'en-US',
            'page': 1
        }
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        data = response.json()
        
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return TMDB_IMAGE_BASE + poster_path
    except Exception as e:
        st.warning(f"TMDB API error: {e}")
    return None

# ===============================================
# Streamlit UI
# ===============================================
st.title("🍿 Movie Recommender with Beautiful Posters 💕")
st.markdown("Enter a movie you love — get similar ones with **covers**! 🎥✨")

movie_input = st.text_input("Movie name (e.g. Fargo, Star Wars, Toy Story):", "Fargo")

if st.button("Find Similar Movies! 😍"):
    with st.spinner("Finding matches & loading posters... 💭"):
        recs_df, error = get_recommendations(movie_input)
        
        if error:
            st.error(error)
        else:
            st.success(f"If you love **{movie_input}**, you'll probably adore these:")
            
            cols = st.columns(4)  # 4-column grid
            
            for i, (_, row) in enumerate(recs_df.iterrows()):
                col = cols[i % 4]
                with col:
                    year = row.get('year', None)
                    poster = get_poster_url(row['clean_title'], year) or get_poster_url(row['title'], year)
                    
                    if poster:
                        st.image(poster, width="stretch")  # ← Fixed! Responsive & no warning
                    else:
                        st.image("https://via.placeholder.com/300x450.png?text=No+Poster", 
                                 width="stretch")
                    
                    st.markdown(f"**{row['title']}**")