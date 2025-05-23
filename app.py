import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_millsongdata.csv")
    df = df.sample(1000, random_state=42).reset_index(drop=True)
    df.dropna(subset=['text'], inplace=True)
    df['song_clean'] = df['song'].str.lower().str.strip()
    return df

# TF-IDF processing
@st.cache_data
def compute_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    return tfidf_matrix

# Recommendation function
def get_recommendations(title_input, df, tfidf_matrix):
    title_input = title_input.lower().strip()
    matched_songs = df[df['song_clean'].str.contains(title_input)]

    if matched_songs.empty:
        st.warning(f"❌ No songs found matching '{title_input}'. Try another title.")
        return None

    if len(matched_songs) > 1:
        st.info("🎯 Multiple songs found. Please choose:")
        song_options = [f"{i}: {row['song']} by {row['artist']}" for i, row in matched_songs.iterrows()]
        selected = st.selectbox("Select a song:", song_options)
        choice = int(selected.split(":")[0])
    else:
        choice = matched_songs.index[0]

    idx = choice
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:5]

    return df.iloc[top_indices][['artist', 'song', 'link']]

# --- Streamlit App ---
st.set_page_config(page_title="🎧 Song Recommender", layout="centered")

st.title("🎵 Song Recommender")
st.write("Find similar songs based on lyrics!")

# Load data
df = load_data()
tfidf_matrix = compute_tfidf(df)

# User Input
user_input = st.text_input("Enter a song title", "")

if st.button("Get Recommendations"):
    if user_input.strip() == "":
        st.warning("Please enter a song title.")
    else:
        recs = get_recommendations(user_input, df, tfidf_matrix)
        if recs is not None:
            st.success("✅ Recommendations:")
            for i, row in recs.iterrows():
                st.markdown(f"**{row['song']}** by *{row['artist']}*  \n[Lyrics]({row['link']})")
