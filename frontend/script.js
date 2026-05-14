// ── Initialize Lucide Icons ──
lucide.createIcons();

// ── Constants & State ──
const API_BASE = "http://localhost:8000/api";
let currentRecommendations = [];
let allSongs = [];

// ── DOM Elements ──
const searchInput = document.getElementById('search-input');
const autocompleteList = document.getElementById('autocomplete-list');
const searchBtn = document.getElementById('search-btn');
const solarSystem = document.getElementById('solar-system');
const modal = document.getElementById('detail-modal');
const modalBody = document.getElementById('modal-body');
const closeModal = document.getElementById('close-modal');

// ── Initialization ──
initStarfield();
fetchSongList();

// ── Starfield Implementation ──
function initStarfield() {
    const starfield = document.getElementById('starfield');
    const count = 150;
    
    for (let i = 0; i < count; i++) {
        const star = document.createElement('div');
        star.className = 'star';
        
        const size = Math.random() * 2.5 + 1;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const duration = Math.random() * 3 + 2;
        const delay = Math.random() * 5;
        
        star.style.width = `${size}px`;
        star.style.height = `${size}px`;
        star.style.left = `${x}%`;
        star.style.top = `${y}%`;
        star.style.setProperty('--duration', `${duration}s`);
        star.style.animationDelay = `${delay}s`;
        
        starfield.appendChild(star);
    }
}

// ── Autocomplete Search ──
async function fetchSongList() {
    try {
        const response = await fetch(`${API_BASE}/songs`);
        allSongs = await response.json();
    } catch (err) {
        console.error("Failed to fetch song list", err);
    }
}

searchInput.addEventListener('input', () => {
    const val = searchInput.value.toLowerCase();
    autocompleteList.innerHTML = '';
    
    if (!val || val.length < 2) {
        autocompleteList.style.display = 'none';
        return;
    }
    
    const matches = allSongs
        .filter(s => s.toLowerCase().includes(val))
        .slice(0, 8);
    
    if (matches.length > 0) {
        matches.forEach(m => {
            const div = document.createElement('div');
            div.className = 'autocomplete-item';
            div.textContent = m;
            div.onclick = () => {
                searchInput.value = m;
                autocompleteList.style.display = 'none';
                getRecommendations(m);
            };
            autocompleteList.appendChild(div);
        });
        autocompleteList.style.display = 'block';
    } else {
        autocompleteList.style.display = 'none';
    }
});

document.addEventListener('click', (e) => {
    if (e.target !== searchInput) {
        autocompleteList.style.display = 'none';
    }
});

searchBtn.onclick = () => {
    if (searchInput.value) {
        getRecommendations(searchInput.value);
    }
};

// ── Recommendation Engine ──
async function getRecommendations(songOption) {
    showStatus("Mapping the galaxy...", "loading");
    
    try {
        const response = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ song_option: songOption })
        });
        
        if (!response.ok) throw new Error("Galaxy mapping failed");
        
        const data = await response.json();
        currentRecommendations = data;
        renderSolarSystem(songOption, data);
        hideStatus();
    } catch (err) {
        showStatus("Could not find that star system. Try another song.", "error");
    }
}

// ── Solar System Renderer ──
function renderSolarSystem(sourceSong, recommendations) {
    solarSystem.innerHTML = '';
    
    // 1. Render Sun
    const sun = document.createElement('div');
    sun.className = 'sun';
    const sourceTitle = sourceSong.split(' — ')[0];
    sun.innerHTML = `
        <i data-lucide="sun" style="color:white; margin-bottom:8px;"></i>
        <div class="sun-title">${sourceTitle}</div>
    `;
    solarSystem.appendChild(sun);
    lucide.createIcons();

    // 2. Render Orbits and Planets
    const orbitCounts = recommendations.length;
    const baseRadius = 160;
    const radiusIncrement = 80;

    // Cosmic color palette for placeholder planets
    const planetThemes = [
        { color: '#fb7185', glow: 'rgba(251, 113, 133, 0.4)' }, // Rose
        { color: '#38bdf8', glow: 'rgba(56, 189, 248, 0.4)' },  // Sky
        { color: '#4ade80', glow: 'rgba(74, 222, 128, 0.4)' },  // Emerald
        { color: '#fbbf24', glow: 'rgba(251, 191, 36, 0.4)' },  // Amber
        { color: '#f472b6', glow: 'rgba(244, 114, 182, 0.4)' }   // Pink
    ];

    recommendations.forEach((rec, index) => {
        const radius = baseRadius + (index * radiusIncrement);
        const duration = 20 + (index * 12); 
        const startAngle = Math.random() * 360; // Randomize starting position
        const theme = planetThemes[index % planetThemes.length];
        
        // Orbit Path
        const orbit = document.createElement('div');
        orbit.className = 'orbit';
        orbit.style.width = `${radius * 2}px`;
        orbit.style.height = `${radius * 2}px`;
        solarSystem.appendChild(orbit);
        
        // Planet Container
        const container = document.createElement('div');
        container.className = 'planet-container';
        container.style.setProperty('--duration', `${duration}s`);
        container.style.setProperty('--start-angle', `${startAngle}deg`);
        
        // Planet itself
        const planet = document.createElement('div');
        planet.className = 'planet';
        planet.style.setProperty('--planet-color', theme.color);
        planet.style.setProperty('--planet-glow', theme.glow);
        
        // Counter-rotation to keep it upright
        planet.style.animation = `rotateOrbit ${duration}s linear infinite reverse`;
        planet.style.setProperty('--start-angle', `${startAngle}deg`);
        
        if (rec.album_art) {
            planet.innerHTML = `<img src="${rec.album_art}" alt="art">`;
        } else {
            planet.innerHTML = `<div class="planet-placeholder"><i data-lucide="music"></i></div>`;
            planet.style.background = `radial-gradient(circle at 30% 30%, ${theme.color}44, ${theme.color}88)`;
        }
        
        const label = document.createElement('div');
        label.className = 'planet-label';
        label.textContent = `${rec.song}`; 
        planet.appendChild(label);
        
        planet.style.left = `calc(50% + ${radius}px - 29px)`;
        planet.style.top = `calc(50% - 29px)`;
        
        planet.onclick = (e) => {
            e.stopPropagation();
            openDetails(rec);
        };
        
        container.appendChild(planet);
        solarSystem.appendChild(container);
    });
    
    lucide.createIcons();
}

// ── Details Modal ──
function openDetails(song) {
    const artHtml = song.album_art 
        ? `<div class="detail-art"><img src="${song.album_art}" alt="art"></div>`
        : `<div class="detail-art" style="background:#222; display:flex; align-items:center; justify-content:center; font-size:3rem;">🎵</div>`;
    
    const previewHtml = song.preview_url
        ? `
        <div class="preview-player">
            <p class="preview-label">30-second Preview</p>
            <audio controls src="${song.preview_url}"></audio>
        </div>`
        : `<p style="margin-top:20px; color:rgba(255,255,255,0.2); font-size:0.8rem;">No audio preview available</p>`;

    const spotifyLink = song.spotify_url
        ? `<a href="${song.spotify_url}" target="_blank" class="btn-primary"><i data-lucide="play"></i> Open in Spotify</a>`
        : '';

    modalBody.innerHTML = `
        <div class="song-detail-header">
            ${artHtml}
            <div class="detail-info">
                <div class="card-rank">MATCH SCORE: ${Math.round(song.similarity * 100)}%</div>
                <h2>${song.song}</h2>
                <p>${song.artist}</p>
                <div class="popularity-meter">
                    <div class="popularity-fill" style="width: ${song.popularity}%"></div>
                </div>
                <div style="font-size:0.65rem; color:rgba(255,255,255,0.3); margin-top:5px;">POPULARITY: ${song.popularity}</div>
            </div>
        </div>
        ${previewHtml}
        <div class="cta-buttons">
            ${spotifyLink}
        </div>
    `;
    
    modal.classList.remove('hidden');
    lucide.createIcons();
}

closeModal.onclick = () => modal.classList.add('hidden');
window.onclick = (e) => { if (e.target === modal) modal.classList.add('hidden'); };

// ── Helpers ──
function showStatus(msg, type) {
    const el = document.getElementById('status-message');
    el.textContent = msg;
    el.className = `status-${type}`;
    el.classList.remove('hidden');
}

function hideStatus() {
    document.getElementById('status-message').classList.add('hidden');
}
