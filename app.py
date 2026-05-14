import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎧 VibeMatch · Song Recommender",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Root & Body ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Animated Gradient Background ── */
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

/* ── Remove Streamlit default header/footer ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Main container ── */
.main .block-container {
    max-width: 780px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* ── Hero Section ── */
.hero-wrapper {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.25));
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 50px;
    padding: 6px 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

.hero-title {
    font-size: 3.4rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #818cf8 80%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: rgba(200,210,255,0.55);
    margin: 0 0 2.2rem 0;
    letter-spacing: 0.02em;
}

/* ── Search Box ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    border-radius: 16px !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    padding: 4px 8px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

[data-testid="stSelectbox"] > div > div:hover,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: rgba(139,92,246,0.8) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15), 0 8px 32px rgba(139,92,246,0.15) !important;
}

/* ── Search label ── */
[data-testid="stSelectbox"] label {
    color: rgba(200,210,255,0.75) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.85rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #7c3aed, #4f46e5, #2563eb) !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    margin-top: 0.5rem;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
    background: linear-gradient(135deg, #8b5cf6, #6366f1, #3b82f6) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Section divider ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2.5rem 0 1.5rem 0;
}

.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
}

.section-header-text {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(167,139,250,0.8);
    white-space: nowrap;
}

/* ── Song Cards ── */
.song-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    animation: cardSlideIn 0.5s ease forwards;
    opacity: 0;
}

.song-card:nth-child(1) { animation-delay: 0.05s; }
.song-card:nth-child(2) { animation-delay: 0.12s; }
.song-card:nth-child(3) { animation-delay: 0.19s; }
.song-card:nth-child(4) { animation-delay: 0.26s; }
.song-card:nth-child(5) { animation-delay: 0.33s; }

@keyframes cardSlideIn {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.song-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #7c3aed, #3b82f6);
    border-radius: 20px 0 0 20px;
}

.song-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(139,92,246,0.35);
    transform: translateX(6px);
    box-shadow: -4px 8px 32px rgba(139,92,246,0.15), 0 4px 20px rgba(0,0,0,0.3);
}

.card-rank {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(139,92,246,0.7);
    margin-bottom: 0.35rem;
}

.card-song-name {
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 0.2rem 0;
    line-height: 1.3;
}

.card-artist {
    font-size: 0.9rem;
    font-weight: 400;
    color: rgba(200,210,255,0.55);
    margin-bottom: 0.8rem;
}

.card-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #818cf8;
    text-decoration: none;
    padding: 4px 12px;
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 50px;
    transition: all 0.25s ease;
    background: rgba(129,140,248,0.08);
}

.card-link:hover {
    background: rgba(129,140,248,0.2);
    border-color: rgba(129,140,248,0.6);
    color: #c4b5fd;
    text-decoration: none;
}

/* ── Warning / Info ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    background: rgba(139,92,246,0.08) !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #a78bfa !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(139,92,246,0.4);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,0.7); }

/* ── Footer ── */
.custom-footer {
    text-align: center;
    margin-top: 3.5rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 0.78rem;
    color: rgba(200,210,255,0.25);
    letter-spacing: 0.05em;
}

/* ── Floating orbs (decorative) ── */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
    animation: floatOrb 10s ease-in-out infinite;
}

.orb-1 {
    width: 400px; height: 400px;
    background: rgba(124,58,237,0.12);
    top: -100px; right: -100px;
}

.orb-2 {
    width: 350px; height: 350px;
    background: rgba(59,130,246,0.1);
    bottom: -80px; left: -80px;
    animation-delay: -5s;
}

@keyframes floatOrb {
    0%, 100% { transform: translate(0, 0); }
    50%       { transform: translate(20px, 20px); }
}
</style>

<!-- Decorative orbs -->
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
    tfidf_matrix = vectorizer.fit_transform(_df['text'])
    return tfidf_matrix


@st.cache_data
def get_song_options(_df):
    """Build autocomplete options: 'Song Title — Artist'"""
    return [f"{row['song']} — {row['artist']}" for _, row in _df.iterrows()]


# ─── Recommendation Logic ─────────────────────────────────────────────────────
def get_recommendations(selected_option, df, tfidf_matrix):
    # Parse selected option
    parts = selected_option.split(" — ", 1)
    song_title = parts[0].strip()

    matched = df[df['song'].str.lower().str.strip() == song_title.lower().strip()]
    if matched.empty:
        matched = df[df['song_clean'].str.contains(song_title.lower().strip())]

    if matched.empty:
        return None, None

    idx = matched.index[0]
    selected_song_info = df.iloc[idx]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:5]

    return df.iloc[top_indices][['artist', 'song', 'link']], selected_song_info


# ─── Hero Section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">🎵 AI-Powered</div>
    <h1 class="hero-title">VibeMatch</h1>
    <p class="hero-sub">Discover songs that feel like your favourite ones.</p>
</div>
""", unsafe_allow_html=True)


# ─── Load Data ────────────────────────────────────────────────────────────────
with st.spinner("Loading the music library..."):
    df = load_data()
    tfidf_matrix = compute_tfidf(df)
    song_options = get_song_options(df)


# ─── Search Bar (Autocomplete) ────────────────────────────────────────────────
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
            recs, seed = get_recommendations(selected_song, df, tfidf_matrix)

        if recs is None:
            st.error("😔  Couldn't find that song in the library. Try another one.")
        else:
            # Section header
            st.markdown("""
            <div class="section-header">
                <div class="section-header-line"></div>
                <div class="section-header-text">✦ Your Recommendations</div>
                <div class="section-header-line"></div>
            </div>
            """, unsafe_allow_html=True)

            # Cards
            for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                link_html = f'<a class="card-link" href="{row["link"]}" target="_blank">♪ View Lyrics</a>' if pd.notna(row.get("link")) and str(row.get("link")).startswith("http") else ""
                st.markdown(f"""
                <div class="song-card">
                    <div class="card-rank">#{rank} Match</div>
                    <div class="card-song-name">{row['song']}</div>
                    <div class="card-artist">{row['artist']}</div>
                    {link_html}
                </div>
                """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    VibeMatch · Built with Streamlit &amp; scikit-learn · Song data from Spotify Million Songs Dataset
</div>
""", unsafe_allow_html=True)
