import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import streamlit as st
# Load dataset
df = pd.read_csv("destinations.csv")
# Create user-item matrix
user_item_matrix = df.pivot_table(index='user', columns='destination', values='rating').fillna(0)
# Normalize data
scaler = StandardScaler()
user_item_scaled = scaler.fit_transform(user_item_matrix)
# Compute similarity between users
similarity = cosine_similarity(user_item_scaled)
similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
# Recommendation logic
def recommend_destinations(user_name, top_n=3):
    if user_name not in similarity_df.index:
        return ["User not found"]
    # Get similar users
    similar_users = similarity_df[user_name].sort_values(ascending=False)[1:]
  # Weighted recommendation
    recommendations = {}
    for other_user, sim_score in similar_users.items():
        other_ratings = user_item_matrix.loc[other_user]
        for destination, rating in other_ratings.items():
            if user_item_matrix.loc[user_name, destination] == 0:  # User hasn't rated it
                if destination not in recommendations:
                    recommendations[destination] = 0
                recommendations[destination] += rating * sim_score
  # Sort and return top destinations
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [dest for dest, score in sorted_recommendations[:top_n]]
# Streamlit UI
st.title("🌍 Travel Recommendation System")
user_input = st.selectbox("Select a user", df['user'].unique())
if st.button("Recommend"):
    results = recommend_destinations(user_input)
    st.write("Top recommended destinations:")
    for i, place in enumerate(results, 1):
        st.write(f"{i}. {place}")
