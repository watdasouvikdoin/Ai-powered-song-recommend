# VibeMatch — AI-Powered Song Recommender

VibeMatch is a content-based music recommendation web app that takes a song you already like and finds five others with a similar feel. It runs entirely in the browser via Streamlit, requires no API keys or external music service accounts, and works off a local dataset of roughly 57,000 songs pulled from the Spotify Million Songs Dataset.

The core idea is simple: song lyrics carry a lot of information about mood, theme, and writing style. Two songs that share lyrical DNA — similar imagery, vocabulary, or emotional register — tend to feel related even when they sit in different genres. This app leans into that by treating lyrics as text, converting them into numerical vectors using TF-IDF, and then ranking how close every other song in the dataset is to the one you picked using cosine similarity.

---

## How It Works

When you launch the app, it loads the CSV dataset and randomly samples 5,000 songs from it (for speed; the full dataset is much larger). It then builds a TF-IDF matrix — this is where each song's lyrics get converted into a weighted bag-of-words representation, with common filler words stripped out via a standard English stop-word list and the vocabulary capped at the 5,000 most meaningful terms.

When you pick a song and hit the button, the app computes the cosine similarity between your chosen song's TF-IDF vector and every other song's vector. Cosine similarity measures the angle between two vectors in high-dimensional space — a score of 1 means the lyrics are nearly identical, while a score near 0 means they share almost nothing in common. The top five closest matches (excluding the song itself) are returned and displayed as cards.

Both the data loading step and the TF-IDF computation are cached with `@st.cache_data`, so the expensive work only runs once per session.

---

## Features

- Autocomplete search bar that shows songs in "Song Title — Artist" format, making it easy to pick the right version when multiple artists share a song name
- Top 5 recommendations displayed as animated cards with the song name, artist, a rank label, and a direct link to the lyrics page when one is available in the dataset
- Fully custom UI built with injected CSS — animated gradient background, glassmorphism-style cards, floating decorative orbs, custom scrollbar, and a clean sans-serif font (Outfit via Google Fonts)
- Graceful handling of edge cases: if a song title can't be found by exact match, it falls back to a substring search before giving up

---

## Project Structure

```
Ai-powered-song-recommend/
├── app.py                  # Main Streamlit application
├── spotify_millsongdata.csv  # Song dataset (artist, song title, lyrics text, link)
├── requirements.txt        # Python dependencies
├── app.py.ipynb            # Notebook version used during development
└── Untitled.ipynb          # Scratch notebook for early exploration
```

---

## Getting Started

**Prerequisites:** Python 3.9 or above.

Clone the repository and install dependencies:

```bash
git clone https://github.com/watdasouvikdoin/Ai-powered-song-recommend.git
cd Ai-powered-song-recommend
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`. The first load takes a few seconds while the dataset is read and the TF-IDF matrix is built; after that, searches are near-instant.

---

## Dependencies

```
streamlit>=1.30.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

No other packages are needed. TF-IDF and cosine similarity both come from scikit-learn's `feature_extraction.text` and `metrics.pairwise` modules respectively.

---

## Dataset

The app uses `spotify_millsongdata.csv`, which contains four columns: `artist`, `song`, `link`, and `text` (the lyrics). The dataset originates from the Spotify Million Song Dataset. At runtime, 5,000 rows are sampled randomly with a fixed seed (`random_state=42`) to keep memory usage and startup time reasonable. Any rows missing a `text` value are dropped before the TF-IDF step.

---

## Limitations

Since recommendations are based purely on lyrical content, the system doesn't account for audio features like tempo, key, instrumentation, or genre. Two songs that sound completely different can end up recommended together if their lyrics share similar themes. This is a known tradeoff of content-based text filtering versus audio-based or collaborative filtering approaches.

The 5,000-song sample also means that results will vary slightly across sessions if you restart the app, since the sample is drawn fresh each time (though the seed keeps it consistent within a session).

---

## Tech Stack

- **Streamlit** — web app framework
- **pandas** — data loading and manipulation
- **scikit-learn** — TF-IDF vectorization and cosine similarity
- **Custom CSS** — all styling, animations, and layout injected via `st.markdown`

---

Built by [Souvik](https://github.com/watdasouvikdoin).
