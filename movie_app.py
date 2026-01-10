import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ===============================================
# Load data & train model (runs once thanks to cache)
# ===============================================
@st.cache_resource
def load_and_train():
    # Load ratings
    ratings = pd.read_csv('ml-100k/ua.base', 
                          sep='\t', 
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # Create matrix
    user_item_matrix = ratings.pivot(index='user_id', 
                                     columns='item_id', 
                                     values='rating').fillna(0)
    
    # SVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    matrix_factorized = svd.fit_transform(user_item_matrix)
    predicted_ratings = np.dot(matrix_factorized, svd.components_)
    
    # Load movies
    movies = pd.read_csv('ml-100k/u.item',
                         sep='|',
                         encoding='latin-1',
                         header=None,
                         usecols=[0, 1],
                         names=['item_id', 'title'])
    
    return user_item_matrix, predicted_ratings, movies

user_item_matrix, predicted_ratings, movies = load_and_train()

# ===============================================
# Recommendation function (same as before)
# ===============================================
def recommend_for_user(user_id, n=10):
    if user_id not in user_item_matrix.index:
        return None, "User not found! Pick between 1 and 943 😊"
    
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_preds = predicted_ratings[user_idx]
    
    preds_series = pd.Series(user_preds, index=user_item_matrix.columns)
    already_rated = user_item_matrix.iloc[user_idx] > 0
    unseen_preds = preds_series[~already_rated]
    
    top_n_ids = unseen_preds.sort_values(ascending=False).head(n).index.tolist()
    
    top_titles = movies[movies['item_id'].isin(top_n_ids)]['title'].tolist()
    return top_titles, None

# ===============================================
# Streamlit UI
# ===============================================
st.title("🍿 Our Movie Recommender 💕")
st.markdown("Built together by **you & your girlfriend** 🫶")

st.write("Choose a user ID (1–943) and see what movies they might love next!")

user_id = st.selectbox(
    "Select User ID",
    options=range(1, 944),
    index=195  # default to 196 (since we tested it)
)

if st.button("Get Recommendations! 🎬"):
    with st.spinner("Thinking about perfect movies for this user... 💭"):
        recs, error = recommend_for_user(user_id, n=10)
        
    if error:
        st.error(error)
    else:
        st.success(f"Here are the top 10 recommendations for User {user_id}:")
        for i, movie in enumerate(recs, 1):
            st.markdown(f"**{i}.** {movie}")

# Cute footer
st.markdown("---")
st.caption("Made with love on January 10, 2026 💗 | Using collaborative filtering magic")