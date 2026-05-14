from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
from typing import List, Optional
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Spotify Init ─────────────────────────────────────────────────────────────
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

# ─── Data & ML Logic ─────────────────────────────────────────────────────────
df = pd.DataFrame()
tfidf_matrix = None
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

def load_data():
    global df, tfidf_matrix
    if os.path.exists("spotify_millsongdata.csv"):
        df = pd.read_csv("spotify_millsongdata.csv")
        df = df.sample(5000, random_state=42).reset_index(drop=True)
        df.dropna(subset=['text'], inplace=True)
        df['song_clean'] = df['song'].str.lower().str.strip()
        tfidf_matrix = vectorizer.fit_transform(df['text'])
    else:
        print("Dataset not found!")

load_data()

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

# ─── Models ──────────────────────────────────────────────────────────────────
class SongRecommendation(BaseModel):
    artist: str
    song: str
    link: Optional[str]
    album_art: Optional[str]
    preview_url: Optional[str]
    spotify_url: Optional[str]
    popularity: int
    similarity: float

class RecommendRequest(BaseModel):
    song_option: str

# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/api/songs")
async def get_songs():
    """Returns a list of songs for autocomplete."""
    options = [f"{row['song']} — {row['artist']}" for _, row in df.iterrows()]
    return options

@app.post("/api/recommend", response_model=List[SongRecommendation])
async def recommend(request: RecommendRequest):
    selected_option = request.song_option
    parts = selected_option.split(" — ", 1)
    song_title = parts[0].strip()
    
    matched = df[df['song'].str.lower().str.strip() == song_title.lower()]
    if matched.empty:
        matched = df[df['song_clean'].str.contains(song_title.lower(), regex=False)]
    
    if matched.empty:
        raise HTTPException(status_code=404, detail="Song not found")
    
    idx = matched.index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:5]
    
    recommendations = []
    for i in top_indices:
        row = df.iloc[i]
        sp_data = get_spotify_data(row['song'], row['artist'])
        
        recommendations.append(SongRecommendation(
            artist=row['artist'],
            song=row['song'],
            link=row['link'] if pd.notna(row['link']) else None,
            album_art=sp_data['album_art'] if sp_data else None,
            preview_url=sp_data['preview_url'] if sp_data else None,
            spotify_url=sp_data['spotify_url'] if sp_data else None,
            popularity=sp_data['popularity'] if sp_data else 0,
            similarity=float(sim_scores[i])
        ))
    
    return recommendations

# Serve frontend static files
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
