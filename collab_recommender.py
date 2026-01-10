import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ===============================================
# 1. Load the training ratings (ua.base)
# ===============================================
print("Loading ratings data...")
ratings = pd.read_csv('ml-100k/ua.base', 
                      sep='\t', 
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

# Create user-item rating matrix (fill missing ratings with 0)
user_item_matrix = ratings.pivot(index='user_id', 
                                 columns='item_id', 
                                 values='rating').fillna(0)

print("User-Item matrix shape:", user_item_matrix.shape)

# ===============================================
# 2. Apply Truncated SVD (simple collaborative filtering)
# ===============================================
print("Training SVD model...")
svd = TruncatedSVD(n_components=50, random_state=42)
matrix_factorized = svd.fit_transform(user_item_matrix)

# Reconstruct predicted ratings
predicted_ratings = np.dot(matrix_factorized, svd.components_)

print("Model trained! ✓")

# ===============================================
# 3. Recommendation function
# ===============================================
def recommend_for_user(user_id, n=10):
    """
    Recommend top N unseen movies for a given user_id
    """
    if user_id not in user_item_matrix.index:
        return ["User ID not found! Please try between 1 and 943 😊"]

    # Get the row index of this user
    user_idx = user_item_matrix.index.get_loc(user_id)
    
    # Get all predicted ratings for this user
    user_preds = predicted_ratings[user_idx]
    
    # Convert to pandas Series for easy sorting
    preds_series = pd.Series(user_preds, index=user_item_matrix.columns)
    
    # Filter out movies the user has already rated (>0)
    already_rated = user_item_matrix.iloc[user_idx] > 0
    unseen_preds = preds_series[~already_rated]
    
    # Get top N highest predicted ratings
    top_n_ids = unseen_preds.sort_values(ascending=False).head(n).index.tolist()
    
    # Load movie titles (FIXED loading!)
    movies = pd.read_csv('ml-100k/u.item',
                         sep='|',
                         encoding='latin-1',
                         header=None,                # Important: no header in file
                         usecols=[0, 1],             # Only columns 0 (id) and 1 (title)
                         names=['item_id', 'title']) # Name them ourselves
    
    # Get titles for the recommended movie ids
    top_titles = movies[movies['item_id'].isin(top_n_ids)]['title'].tolist()
    
    return top_titles

# ===============================================
# 4. Test it!
# ===============================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("   🎬 MOVIE RECOMMENDER - COLLABORATIVE FILTERING   ")
    print("="*50 + "\n")
    
    # Example recommendations
    test_user = 196  # You can change this number (1-943)
    num_recs = 8
    
    print(f"Generating top {num_recs} recommendations for user {test_user}...\n")
    
    recommendations = recommend_for_user(test_user, n=num_recs)
    
    print(f"Top {num_recs} recommendations for user {test_user}:")
    print("-"*60)
    if isinstance(recommendations[0], str) and "not found" in recommendations[0]:
        print(recommendations[0])
    else:
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")
    print("-"*60)