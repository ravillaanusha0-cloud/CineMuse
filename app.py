    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(9,10,16,.72); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
p, label, .stMarkdown { color: #e1e3ee; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11121d 0%, #0b0c13 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h1 { font-size: 1.8rem; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #d9daea; }
div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #141622 !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,.13) !important;
    border-radius: 13px !important;
}
div.stButton > button {
    border: 1px solid rgba(255,255,255,.14);
    background: #191b29;
    color: #f5f5fb;
    border-radius: 12px;
    min-height: 2.55rem;
    font-weight: 600;
    transition: all .18s ease;
}
div.stButton > button:hover {
    border-color: var(--pink);
    color: white;
    background: linear-gradient(100deg, rgba(255,79,145,.22), rgba(154,107,255,.22));
    transform: translateY(-1px);
}
.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 25px;
    padding: 2.1rem 2.2rem;
    margin: .5rem 0 1.7rem 0;
    background:
      radial-gradient(circle at 78% 22%, rgba(255,79,145,.30), transparent 25%),
      radial-gradient(circle at 88% 80%, rgba(122,91,255,.28), transparent 30%),
      linear-gradient(120deg, #191a2a 0%, #171323 52%, #241329 100%);
    box-shadow: 0 20px 65px rgba(0,0,0,.23);
}
.hero-kicker {
    color: #ff8fbb;
    letter-spacing: .20em;
    text-transform: uppercase;
    font-size: .75rem;
    font-weight: 700;
}
.hero-title {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.1rem, 4vw, 3.7rem);
    font-weight: 700;
    line-height: 1.08;
    margin: .6rem 0 .7rem;
    max-width: 700px;
}
.hero-copy { color: #c7c7d8; font-size: 1.02rem; max-width: 620px; line-height: 1.65; }
.hero-orb {
    float: right;
    font-size: 5.5rem;
    padding: .6rem 1rem;
.pill {
    display: inline-block; border: 1px solid rgba(255,111,165,.32);
    background: rgba(255,79,145,.10); color: #ff9bc0;
    border-radius: 999px; padding: .25rem .6rem; font-size: .72rem; margin-top: .25rem;
}
div[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: rgba(255,255,255,.10); }
footer { visibility: hidden; }
@media (max-width: 700px) {
    .hero { padding: 1.4rem; }
    .hero-orb { font-size: 3rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------- MOVIE DATA ----------
movies = pd.DataFrame([
    {"title": "Inception", "year": 2010, "genre": "Sci-Fi", "description": "Dreams, mystery, action and mind-bending science fiction", "icon": "🌀"},
    {"title": "Interstellar", "year": 2014, "genre": "Sci-Fi", "description": "Space, planets, time travel, science and family", "icon": "🚀"},
    {"title": "The Dark Knight", "year": 2008, "genre": "Action", "description": "Superhero, crime, suspense and action", "icon": "🦇"},
    {"title": "La La Land", "year": 2016, "genre": "Romance", "description": "Love, music, dreams, dancing and romance", "icon": "🎹"},
    {"title": "The Shawshank Redemption", "year": 1994, "genre": "Drama", "description": "Friendship, hope, prison and human courage", "icon": "🎞️"},
    {"title": "Spirited Away", "year": 2001, "genre": "Animation", "description": "Fantasy, magic, adventure and animated storytelling", "icon": "✨"},
    {"title": "The Martian", "year": 2015, "genre": "Sci-Fi", "description": "Mars, space, survival, science and adventure", "icon": "🪐"},
    {"title": "The Godfather", "year": 1972, "genre": "Crime", "description": "Family, crime, loyalty and powerful drama", "icon": "🕴️"},
    {"title": "Pride and Prejudice", "year": 2005, "genre": "Romance", "description": "Love, family, relationships and period drama", "icon": "🌹"},
    {"title": "Jumanji", "year": 1995, "genre": "Adventure", "description": "Adventure, fantasy, games and comedy", "icon": "🌴"},
    {"title": "The Conjuring", "year": 2013, "genre": "Horror", "description": "Horror, mystery, ghosts and supernatural suspense", "icon": "👁️"},
    {"title": "Toy Story", "year": 1995, "genre": "Animation", "description": "Friendship, toys, comedy and family adventure", "icon": "🧸"},
])

movies["features"] = movies["genre"] + " " + movies["description"]
vectorizer = TfidfVectorizer(stop_words="english")
movie_vectors = vectorizer.fit_transform(movies["features"])
similarity = cosine_similarity(movie_vectors)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🎬 CineMuse")
    st.caption("DISCOVER  ·  WATCH  ·  REPEAT")
    st.markdown("---")
    page = st.radio(
        "YOUR SPACE",
        ["Home", "Discover", "Recommendations", "My Watchlist"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**YOUR MOVIE MOOD**")
    st.caption("Pick a genre on the main page to filter the collection.")
    st.markdown(
        "<div style='padding:14px;border-radius:14px;background:linear-gradient(135deg,#21172e,#171421);border:1px solid #392943;color:#c7c3d9;'>"
        "<span style='color:#ff8fbb;font-weight:700;'>GOOD MOVIES.</span><br>"
        "Better moods. New worlds. Stories that stay with you. ✦</div>",
        unsafe_allow_html=True,
    )

# ---------- HEADER / HERO ----------
left, right = st.columns([5, 1])
with left:
    st.markdown("<div class='section-eyebrow'>YOUR PERSONAL MOVIE UNIVERSE</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Grotesk;font-size:2.25rem;font-weight:700;line-height:1.1;color:#fff;'>Cine<span style='color:#ff4f91;'>Muse</span></div>", unsafe_allow_html=True)
with right:
    st.markdown("<div style='text-align:right;font-size:2.4rem;padding-top:.2rem;'>✦</div>", unsafe_allow_html=True)

if page == "Home":
    st.markdown("""
    <div class="hero">
<div class="hero-orb">🎬</div>
      <div class="hero-kicker">A different kind of movie night</div>
      <div class="hero-title">Find stories that feel like <span style="color:#ff77aa;">you.</span></div>
      <div class="hero-copy">Explore your next favourite film, follow a feeling, and let smart recommendations lead you to your next watch.</div>
    </div>
    """, unsafe_allow_html=True)
elif page == "Discover":
    st.markdown("<div class='hero'><div class='hero-kicker'>Explore the collection</div><div class='hero-title'>Your next obsession starts here.</div><div class='hero-copy'>Search the collection or choose a genre to find something worth watching.</div></div>", unsafe_allow_html=True)
elif page == "Recommendations":
    st.markdown("<div class='hero'><div class='hero-kicker'>Made around your taste</div><div class='hero-title'>One favourite. Many new worlds.</div><div class='hero-copy'>Choose a film you love and CineMuse will suggest similar stories using content similarity.</div></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='hero'><div class='hero-kicker'>Saved for later</div><div class='hero-title'>Your personal watchlist.</div><div class='hero-copy'>All the stories you want to come back to, in one place.</div></div>", unsafe_allow_html=True)

# ---------- GENRES AT TOP: HIGH VISIBILITY ----------
if page in ["Home", "Discover"]:
    st.markdown("<div class='section-eyebrow'>EXPLORE BY MOOD</div><div class='section-title'>Choose your genre</div><div class='genre-note'>Tap a genre to instantly filter the movies below.</div>", unsafe_allow_html=True)
    genres = ["All"] + sorted(movies["genre"].unique().tolist())
    genre_icons = {"All": "✦", "Action": "⚡", "Adventure": "🧭", "Animation": "✨", "Crime": "🕶️", "Drama": "🎭", "Horror": "🌑", "Romance": "💗", "Sci-Fi": "🚀"}
    genre_cols = st.columns(len(genres))
    for i, genre in enumerate(genres):
        is_selected = st.session_state.selected_genre == genre
        with genre_cols[i]:
            st.markdown("<div class='genre-chip genre-active'>" if is_selected else "<div class='genre-chip'>", unsafe_allow_html=True)
            if st.button(f"{'✓  ' if is_selected else ''}{genre_icons.get(genre, '✦')}  {genre}", key=f"genre_{genre}", use_container_width=True):
                st.session_state.selected_genre = genre
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    search = st.text_input("⌕  Search the collection", placeholder="Search by movie title, genre, or mood...")
    filtered = movies.copy()
    if st.session_state.selected_genre != "All":
        filtered = filtered[filtered["genre"] == st.session_state.selected_genre]
    if search:
        mask = (
            filtered["title"].str.contains(search, case=False, na=False)
            | filtered["genre"].str.contains(search, case=False, na=False)
            | filtered["description"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]

    st.markdown("---")
    st.markdown("<div class='section-eyebrow'>CURATED FOR YOUR NEXT MOVIE NIGHT</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{'Trending Movies' if page == 'Home' else 'Discover Movies'} <span style='font-size:.95rem;color:#a5a9bd;font-weight:500;'>· {len(filtered)} titles</span></div>", unsafe_allow_html=True)
    if filtered.empty:
        st.info("No movies found. Try another search or genre.")
    else:
        cols = st.columns(3)
        for i, (_, movie) in enumerate(filtered.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="movie-card">
                  <div class="movie-icon">{movie['icon']}</div>
                  <div class="movie-title">{movie['title']}</div>
                  <div class="movie-meta">{movie['year']} &nbsp;·&nbsp; {movie['genre']}</div>
                  <div class="movie-description">{movie['description']}</div>
                  <span class="pill">CINEMUSE PICK</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button("＋  Add to watchlist", key=f"save_{movie['title']}", use_container_width=True):
                    if movie["title"] not in st.session_state.watchlist:
                        st.session_state.watchlist.append(movie["title"])
                        st.success(f"Added {movie['title']} to your watchlist.")
                    else:
                        st.info("This movie is already in your watchlist.")

# ---------- RECOMMENDATIONS ----------
if page == "Recommendations":
    st.markdown("<div class='section-eyebrow'>CONTENT-BASED DISCOVERY</div><div class='section-title'>Recommended for you</div>", unsafe_allow_html=True)
    selected_movie = st.selectbox("Choose a movie you enjoy", movies["title"].tolist())
    movie_index = movies.index[movies["title"] == selected_movie][0]
    scores = sorted(enumerate(similarity[movie_index]), key=lambda x: x[1], reverse=True)
    recommended_indices = [i for i, score in scores if i != movie_index][:5]
    st.caption("Suggestions are based on similarities between movie genres and descriptions.")
    cols = st.columns(3)
    for pos, index in enumerate(recommended_indices):
        movie = movies.iloc[index]
        with cols[pos % 3]:
            st.markdown(f"""
            <div class="movie-card">
              <div class="movie-icon">{movie['icon']}</div>
              <div class="movie-title">{movie['title']}</div>
              <div class="movie-meta">{movie['year']} &nbsp;·&nbsp; {movie['genre']}</div>
              <div class="movie-description">{movie['description']}</div>
              <span class="pill">SIMILAR STORY</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("＋  Add to watchlist", key=f"rec_{movie['title']}", use_container_width=True):
                if movie["title"] not in st.session_state.watchlist:
                    st.session_state.watchlist.append(movie["title"])
                    st.success("Movie added to your watchlist.")
                else:
                    st.info("This movie is already in your watchlist.")

# ---------- WATCHLIST ----------
if page == "My Watchlist":
    st.markdown("<div class='section-eyebrow'>YOUR SAVED COLLECTION</div><div class='section-title'>❤️ My Watchlist</div>", unsafe_allow_html=True)
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Explore movies and add titles you want to watch.")
    else:
        for title in list(st.session_state.watchlist):
            movie = movies[movies["title"] == title].iloc[0]
            with st.container():
                st.markdown(f"""
                <div class="movie-card" style="min-height:0;">
                  <div class="movie-title">{movie['icon']} &nbsp; {movie['title']}</div>
                  <div class="movie-meta">{movie['year']} &nbsp;·&nbsp; {movie['genre']}</div>
                  <div class="movie-description">{movie['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Remove from watchlist", key=f"remove_{title}"):
                    st.session_state.watchlist.remove(title)
                    st.rerun()

st.markdown("<div style='text-align:center;color:#777b91;font-size:.78rem;padding:2rem 0 0;'>CINEMUSE &nbsp; ✦ &nbsp; DISCOVER SOMETHING THAT MOVES YOU</div>", unsafe_allow_html=True)

