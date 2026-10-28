
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMuse",
    page_icon="🎬",
    layout="wide"
)

# Dark cherry-pink design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #100d14, #09090b);
    color: #f7f3f6;
}
section[data-testid="stSidebar"] {
    background: #151019;
    border-right: 1px solid #30232e;
}
h1, h2, h3 {
    color: #fff0f5 !important;
}
p, label {
    color: #ded4df;
}
div.stButton > button {
    background: #ed477d;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 16px;
}
div.stButton > button:hover {
    background: #ff6595;
    color: white;
}
.movie-card {
    background: #19141e;
    border: 1px solid #392535;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 10px;
    min-height: 165px;
}
.movie-year {
    color: #ff8eb3;
    font-size: 13px;
}
.movie-description {
    color: #cfc3d1;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Movie dataset
movies = pd.DataFrame([
    {
        "title": "Inception",
        "year": 2010,
        "genre": "Sci-Fi",
        "description": "Dreams, mystery, action and mind-bending science fiction"
    },
    {
        "title": "Interstellar",
        "year": 2014,
        "genre": "Sci-Fi",
        "description": "Space, planets, time travel, science and family"
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "genre": "Action",
        "description": "Superhero, crime, suspense and action"
    },
    {
        "title": "La La Land",
        "year": 2016,
        "genre": "Romance",
        "description": "Love, music, dreams, dancing and romance"
    },
    {
        "title": "The Shawshank Redemption",
        "year": 1994,
        "genre": "Drama",
        "description": "Friendship, hope, prison and human courage"
    },
    {
        "title": "Spirited Away",
        "year": 2001,
        "genre": "Animation",
        "description": "Fantasy, magic, adventure and animated storytelling"
    },
    {
        "title": "The Martian",
        "year": 2015,
        "genre": "Sci-Fi",
        "description": "Mars, space, survival, science and adventure"
    },
    {
        "title": "The Godfather",
        "year": 1972,
        "genre": "Crime",
        "description": "Family, crime, loyalty and powerful drama"
    },
    {
        "title": "Pride and Prejudice",
        "year": 2005,
        "genre": "Romance",
        "description": "Love, family, relationships and period drama"
    },
    {
        "title": "Jumanji",
        "year": 1995,
        "genre": "Adventure",
        "description": "Adventure, fantasy, games and comedy"
    },
    {
        "title": "The Conjuring",
        "year": 2013,
        "genre": "Horror",
        "description": "Horror, mystery, ghosts and supernatural suspense"
    },
    {
        "title": "Toy Story",
        "year": 1995,
        "genre": "Animation",
        "description": "Friendship, toys, comedy and family adventure"
    }
])

# Prepare recommendation engine
movies["features"] = (
    movies["genre"] + " " + movies["description"]
)

vectorizer = TfidfVectorizer(stop_words="english")
movie_vectors = vectorizer.fit_transform(movies["features"])

similarity = cosine_similarity(movie_vectors)

# Store watchlist during the session
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# Sidebar
st.sidebar.title("🎬 CineMuse")
st.sidebar.caption("Discover • Watch • Review")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Discover", "Recommendations", "My Watchlist"]
)

genres = ["All"] + sorted(movies["genre"].unique().tolist())
selected_genre = st.sidebar.selectbox("Choose a genre", genres)

# Header
st.title("CineMuse")
st.caption("Discover movies that match your mood.")

if page == "Home":
    st.markdown("""
    <div class="movie-card">
        <h2>Find your next favourite movie.</h2>
        <p class="movie-description">
        Explore stories, discover new genres and get recommendations
        based on the movies you enjoy.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Search and filter
if page in ["Home", "Discover"]:
    search = st.text_input("🔎 Search movies", placeholder="Search by title or genre...")

    filtered = movies.copy()

    if selected_genre != "All":
        filtered = filtered[filtered["genre"] == selected_genre]

    if search:
        filtered = filtered[
            filtered["title"].str.contains(search, case=False, na=False)
            | filtered["genre"].str.contains(search, case=False, na=False)
            | filtered["description"].str.contains(search, case=False, na=False)
        ]

    st.subheader("Trending Movies" if page == "Home" else "Discover Movies")

    if filtered.empty:
        st.info("No movies found. Try another search or genre.")
    else:
        cols = st.columns(3)

        for i, (_, movie) in enumerate(filtered.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="movie-card">
                    <h3>{movie['title']}</h3>
                    <div class="movie-year">
                        {movie['year']} · {movie['genre']}
                    </div>
                    <p class="movie-description">
                        {movie['description']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "＋ Add to My List",
                    key=f"save_{movie['title']}"
                ):
                    title = movie["title"]

                    if title not in st.session_state.watchlist:
                        st.session_state.watchlist.append(title)
                        st.success(f"{title} added to your list!")
                    else:
                        st.info("This movie is already in your list.")

# Recommendations
if page == "Recommendations":
    st.subheader("Recommended For You")

    selected_movie = st.selectbox(
        "Choose a movie you enjoy",
        movies["title"].tolist()
    )

    movie_index = movies.index[
        movies["title"] == selected_movie
    ][0]

    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [
        i for i, score in scores if i != movie_index
    ][:5]

    for index in recommended_indices:
        movie = movies.iloc[index]

        st.markdown(f"""
        <div class="movie-card">
            <h3>{movie['title']}</h3>
            <div class="movie-year">
                {movie['year']} · {movie['genre']}
            </div>
            <p class="movie-description">{movie['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("＋ Add to My List", key=f"rec_{movie['title']}"):
            if movie["title"] not in st.session_state.watchlist:
                st.session_state.watchlist.append(movie["title"])
                st.success("Movie added to your list!")
            else:
                st.info("Already in your list.")

# Watchlist
if page == "My Watchlist":
    st.subheader("❤️ My Watchlist")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add some movies first.")
    else:
        for title in st.session_state.watchlist:
            movie = movies[movies["title"] == title].iloc[0]

            col1, col2 = st.columns([4, 1])

            with col1:
                st.write(f"**{movie['title']}** ({movie['year']})")
                st.caption(movie["genre"])

            with col2:
                if st.button("Remove", key=f"remove_{title}"):
                    st.session_state.watchlist.remove(title)
                    st.rerun()

st.sidebar.divider()
st.sidebar.caption("CineMuse · Your personal movie discovery app")
