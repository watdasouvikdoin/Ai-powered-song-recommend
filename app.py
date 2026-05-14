import streamlit as st
import html as html_lib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎧 VibeMatch · Song Recommender",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Spotify Init ─────────────────────────────────────────────────────────────
@st.cache_resource
def init_spotify():
    try:
        auth = SpotifyClientCredentials(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
        )
        return spotipy.Spotify(auth_manager=auth)
    except Exception:
        return None

sp = init_spotify()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #0a0a0f, #0d0d1a, #080c18, #0f0a1a, #0a1020);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    min-height: 100vh;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.main .block-container { max-width: 780px; padding-top: 3rem; padding-bottom: 4rem; }

/* Hero */
.hero-wrapper { text-align: center; padding: 2rem 0 1.5rem 0; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.25));
    border: 1px solid rgba(139,92,246,0.4); border-radius: 50px;
    padding: 6px 20px; font-size: 0.78rem; font-weight: 500;
    letter-spacing: 0.15em; text-transform: uppercase; color: #a78bfa;
    margin-bottom: 1.2rem; backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 3.4rem; font-weight: 700; line-height: 1.1; margin: 0 0 0.6rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #818cf8 80%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub { font-size: 1.05rem; font-weight: 300; color: rgba(200,210,255,0.55); margin: 0 0 2.2rem 0; }

/* Search */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.35) !important; border-radius: 16px !important;
    color: white !important; font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important; padding: 4px 8px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(139,92,246,0.8) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15), 0 8px 32px rgba(139,92,246,0.15) !important;
}
[data-testid="stSelectbox"] label {
    color: rgba(200,210,255,0.75) !important; font-size: 0.85rem !important;
    font-weight: 500 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important;
}

/* Button */
[data-testid="stButton"] > button {
    width: 100%; padding: 0.85rem 2rem !important; font-size: 1rem !important;
    font-weight: 600 !important; font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.05em !important; border: none !important; border-radius: 14px !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #2563eb) !important;
    color: white !important; cursor: pointer !important; transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important; margin-top: 0.5rem;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
    background: linear-gradient(135deg, #8b5cf6, #6366f1, #3b82f6) !important;
}

/* Section header */
.section-header { display: flex; align-items: center; gap: 10px; margin: 2.5rem 0 1.5rem 0; }
.section-header-line { flex: 1; height: 1px; background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent); }
.section-header-text { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: rgba(167,139,250,0.8); white-space: nowrap; }

/* Cards */
.song-card {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative; overflow: hidden;
    animation: cardSlideIn 0.5s ease forwards; opacity: 0;
}
.song-card:nth-child(1){animation-delay:0.05s} .song-card:nth-child(2){animation-delay:0.12s}
.song-card:nth-child(3){animation-delay:0.19s} .song-card:nth-child(4){animation-delay:0.26s}
.song-card:nth-child(5){animation-delay:0.33s}
@keyframes cardSlideIn { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
.song-card::before {
    content:''; position:absolute; top:0; left:0; width:4px; height:100%;
    background: linear-gradient(180deg, #7c3aed, #3b82f6); border-radius: 20px 0 0 20px;
}
.song-card:hover {
    background: rgba(255,255,255,0.07); border-color: rgba(139,92,246,0.35);
    transform: translateX(6px);
    box-shadow: -4px 8px 32px rgba(139,92,246,0.15), 0 4px 20px rgba(0,0,0,0.3);
}

/* Card inner layout */
.card-inner { display: flex; gap: 1.1rem; align-items: flex-start; }
.card-art { flex-shrink: 0; width: 82px; height: 82px; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(0,0,0,0.5); }
.card-art img { width: 100%; height: 100%; object-fit: cover; display: block; }
.card-art-placeholder {
    flex-shrink: 0; width: 82px; height: 82px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(59,130,246,0.3));
    display: flex; align-items: center; justify-content: center; font-size: 1.8rem;
    border: 1px solid rgba(139,92,246,0.2);
}
.card-content { flex: 1; min-width: 0; }
.card-rank { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(139,92,246,0.7); margin-bottom: 0.25rem; }
.card-song-name { font-size: 1.1rem; font-weight: 600; color: #ffffff; margin: 0 0 0.15rem 0; line-height: 1.3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card-artist { font-size: 0.88rem; color: rgba(200,210,255,0.5); margin-bottom: 0.5rem; }

/* Popularity bar */
.popularity-row { display: flex; align-items: center; gap: 8px; margin-bottom: 0.75rem; }
.popularity-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: rgba(200,210,255,0.3); white-space: nowrap; }
.popularity-bar { flex: 1; height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; }
.popularity-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, #7c3aed, #3b82f6); }

/* Action buttons */
.card-actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.spotify-btn {
    display: inline-flex; align-items: center; gap: 5px; font-size: 0.78rem; font-weight: 600;
    color: #1ed760 !important; text-decoration: none !important; padding: 5px 13px;
    border: 1px solid rgba(30,215,96,0.35); border-radius: 50px;
    background: rgba(30,215,96,0.08); transition: all 0.25s ease; font-family: 'Outfit', sans-serif;
}
.spotify-btn:hover { background: rgba(30,215,96,0.18); border-color: rgba(30,215,96,0.6); }
.lyrics-btn {
    display: inline-flex; align-items: center; gap: 5px; font-size: 0.78rem; font-weight: 500;
    color: #818cf8 !important; text-decoration: none !important; padding: 5px 13px;
    border: 1px solid rgba(129,140,248,0.3); border-radius: 50px;
    background: rgba(129,140,248,0.08); transition: all 0.25s ease; font-family: 'Outfit', sans-serif;
}
.lyrics-btn:hover { background: rgba(129,140,248,0.2); border-color: rgba(129,140,248,0.6); }

/* Audio preview */
.preview-section { margin-top: 1rem; padding-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.06); }
.preview-label { font-size: 0.68rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(200,210,255,0.3); margin-bottom: 6px; }
audio { width: 100%; height: 36px; border-radius: 8px; outline: none; filter: invert(1) hue-rotate(180deg) brightness(0.85); }

/* Spotify connected badge */
.spotify-status {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(30,215,96,0.1); border: 1px solid rgba(30,215,96,0.25);
    border-radius: 50px; padding: 4px 14px; font-size: 0.75rem; font-weight: 500;
    color: #1ed760; margin-bottom: 1.5rem;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #1ed760; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Footer */
.custom-footer { text-align: center; margin-top: 3.5rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.78rem; color: rgba(200,210,255,0.25); }

/* Orbs */
.orb { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; animation: floatOrb 10s ease-in-out infinite; }
.orb-1 { width: 400px; height: 400px; background: rgba(124,58,237,0.12); top: -100px; right: -100px; }
.orb-2 { width: 350px; height: 350px; background: rgba(59,130,246,0.1); bottom: -80px; left: -80px; animation-delay: -5s; }
@keyframes floatOrb { 0%,100%{transform:translate(0,0)} 50%{transform:translate(20px,20px)} }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 3px; }
</style>

<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_millsongdata.csv")
    df = df.sample(5000, random_state=42).reset_index(drop=True)
    df.dropna(subset=['text'], inplace=True)
    df['song_clean'] = df['song'].str.lower().str.strip()
    return df

@st.cache_data
def compute_tfidf(_df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    return vectorizer.fit_transform(_df['text'])

@st.cache_data
def get_song_options(_df):
    return [f"{row['song']} — {row['artist']}" for _, row in _df.iterrows()]

@st.cache_data
def get_spotify_data(song_name: str, artist_name: str):
    if sp is None:
        return None
    try:
        results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
        tracks = results['tracks']['items']
        if not tracks:
            results = sp.search(q=f"{song_name} {artist_name}", type='track', limit=1)
            tracks = results['tracks']['items']
        if tracks:
            t = tracks[0]
            imgs = t['album']['images']
            art = imgs[1]['url'] if len(imgs) > 1 else (imgs[0]['url'] if imgs else None)
            return {
                'album_art': art,
                'preview_url': t['preview_url'],
                'spotify_url': t['external_urls']['spotify'],
                'popularity': t['popularity'],
            }
    except Exception:
        pass
    return None


# ─── Recommendation Logic ─────────────────────────────────────────────────────
def get_recommendations(selected_option, df, tfidf_matrix):
    parts = selected_option.split(" — ", 1)
    song_title = parts[0].strip()
    matched = df[df['song'].str.lower().str.strip() == song_title.lower()]
    if matched.empty:
        matched = df[df['song_clean'].str.contains(song_title.lower(), regex=False)]
    if matched.empty:
        return None
    idx = matched.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:5]
    return df.iloc[top_indices][['artist', 'song', 'link']]


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">🎵 AI-Powered</div>
    <h1 class="hero-title">VibeMatch</h1>
    <p class="hero-sub">Discover songs that feel like your favourite ones.</p>
</div>
""", unsafe_allow_html=True)

# Spotify status badge
if sp is not None:
    st.markdown('<div style="text-align:center"><span class="spotify-status"><span class="status-dot"></span>Spotify Connected</span></div>', unsafe_allow_html=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading music library..."):
    df = load_data()
    tfidf_matrix = compute_tfidf(df)
    song_options = get_song_options(df)

# ─── Search ───────────────────────────────────────────────────────────────────
selected_song = st.selectbox(
    "🔍  Search for a song",
    options=[""] + song_options,
    index=0,
    placeholder="Type a song name or artist...",
)

find_btn = st.button("✦  Find Similar Songs")

# ─── Results ─────────────────────────────────────────────────────────────────
if find_btn:
    if not selected_song:
        st.warning("⚠️  Please select a song from the search bar first.")
    else:
        with st.spinner("Finding your vibes..."):
            recs = get_recommendations(selected_song, df, tfidf_matrix)

        if recs is None:
            st.error("😔  Couldn't find that song. Try another one.")
        else:
            st.markdown("""
            <div class="section-header">
                <div class="section-header-line"></div>
                <div class="section-header-text">✦ Your Recommendations</div>
                <div class="section-header-line"></div>
            </div>""", unsafe_allow_html=True)

            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                sp_data = get_spotify_data(row['song'], row['artist'])

                # Safe-escape text values
                safe_song = html_lib.escape(str(row['song']))
                safe_artist = html_lib.escape(str(row['artist']))

                # Album art / placeholder
                if sp_data and sp_data.get('album_art'):
                    art_html = f'<div class="card-art"><img src="{sp_data["album_art"]}" alt="cover"/></div>'
                else:
                    art_html = '<div class="card-art-placeholder">\U0001f3b5</div>'

                # Popularity bar (single line, no leading whitespace)
                if sp_data:
                    pop = sp_data['popularity']
                    pop_html = f'<div class="popularity-row"><span class="popularity-label">Popularity</span><div class="popularity-bar"><div class="popularity-fill" style="width:{pop}%"></div></div><span class="popularity-label">{pop}</span></div>'
                else:
                    pop_html = ''

                # Action buttons
                actions = ''
                if sp_data and sp_data.get('spotify_url'):
                    actions += f'<a class="spotify-btn" href="{sp_data["spotify_url"]}" target="_blank">&#9654; Open in Spotify</a>'
                if pd.notna(row.get('link')) and str(row.get('link', '')).startswith('http'):
                    actions += f'<a class="lyrics-btn" href="{row["link"]}" target="_blank">&#9835; Lyrics</a>'

                # Audio preview (single line)
                if sp_data and sp_data.get('preview_url'):
                    preview_html = f'<div class="preview-section"><div class="preview-label">&#127911; 30-sec Preview</div><audio controls src="{sp_data["preview_url"]}"></audio></div>'
                else:
                    preview_html = ''

                card = (
                    f'<div class="song-card">'
                    f'<div class="card-inner">{art_html}'
                    f'<div class="card-content">'
                    f'<div class="card-rank">#{rank} Match</div>'
                    f'<div class="card-song-name">{safe_song}</div>'
                    f'<div class="card-artist">{safe_artist}</div>'
                    f'{pop_html}'
                    f'<div class="card-actions">{actions}</div>'
                    f'</div></div>'
                    f'{preview_html}'
                    f'</div>'
                )
                st.markdown(card, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    VibeMatch · Powered by Streamlit, scikit-learn &amp; Spotify API
</div>
""", unsafe_allow_html=True)
