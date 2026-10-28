import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMuse | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 4-SECOND CINEMUSE INTRO ----------
# Show once per browser session; the overlay fades away after four seconds.
if "cinemuse_intro_shown" not in st.session_state:
    st.session_state.cinemuse_intro_shown = True
    st.markdown("""
    <style>
    #cinemuse-intro {
        position: fixed; inset: 0; z-index: 999999;
        display: flex; align-items: center; justify-content: center;
        background: radial-gradient(circle at 50% 40%, #282044 0%, #11101d 48%, #070810 100%);
        animation: cinemuseIntroFade 4s ease forwards;
        pointer-events: none;
    }
    #cinemuse-intro .intro-name {
        font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.7rem, 8vw, 5.8rem);
        font-weight: 800; letter-spacing: -.06em; color: #fff;
        text-shadow: 0 0 32px rgba(255,79,145,.35);
        animation: cinemuseIntroGlow 1.5s ease-in-out infinite alternate;
    }
    #cinemuse-intro .intro-name span { color: #ff4f91; }
    #cinemuse-intro .intro-tagline { margin-top: 14px; text-align: center; color: #c7bfdc;
        font: 600 .78rem 'DM Sans', sans-serif; letter-spacing: .24em; }
    @keyframes cinemuseIntroFade {
        0%, 78% { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; }
    }
    @keyframes cinemuseIntroGlow { from { transform: scale(.98); opacity: .88; } to { transform: scale(1.02); opacity: 1; } }
    @media (prefers-reduced-motion: reduce) { #cinemuse-intro { animation-duration: 4s; } }
    </style>
    <div id="cinemuse-intro"><div><div class="intro-name">Cine<span>Muse</span></div><div class="intro-tagline">YOUR NEXT STORY STARTS HERE</div></div></div>
    """, unsafe_allow_html=True)

# ---------- CINEMATIC THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #090a10;
    --panel: #12141f;
    --panel2: #181b29;
    --text: #f5f5fb;
    --muted: #a5a9bd;
    --pink: #ff4f91;
    --violet: #9a6bff;
    --line: rgba(255,255,255,.10);
}
.stApp {
    background:
      radial-gradient(ellipse at 12% 0%, rgba(119, 69, 205, .18), transparent 34%),
      radial-gradient(ellipse at 95% 10%, rgba(255, 79, 145, .11), transparent 28%),
      var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: rgba(9,10,16,.72); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1450px; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
p, label, .stMarkdown { color: #e1e3ee; }
section[data-testid="stSidebar"] {
    background:
      radial-gradient(ellipse at 10% 0%, rgba(154,107,255,.15), transparent 34%),
      linear-gradient(180deg, #11121d 0%, #0b0c13 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] > div { padding-top: 1.25rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #a9adc0; }
section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #f1effa !important;
    font-size: .77rem;
    font-weight: 800;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .8rem;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .42rem; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border: 1px solid transparent;
    border-radius: 13px;
    padding: .58rem .7rem;
    margin: 0;
    transition: all .18s ease;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.045);
    border-color: rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p { color: #e5e3ef !important; font-weight: 600; }
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
    filter: drop-shadow(0 0 24px rgba(255,79,145,.28));
}
.section-eyebrow {
    color: #ff8fbb; font-size: .72rem; font-weight: 700;
    letter-spacing: .17em; text-transform: uppercase; margin-bottom: .2rem;
}
.section-title { font-size: 1.55rem; font-weight: 700; margin: 0 0 .25rem 0; }
.section-copy { color: var(--muted); font-size: .92rem; margin-bottom: .9rem; }
.genre-note { color: #b8bbcf; font-size: .86rem; margin: .1rem 0 .65rem; }
.genre-active div.stButton > button {
    background: linear-gradient(110deg, #ff4f91, #9a6bff);
    color: white; border-color: transparent;
    box-shadow: 0 5px 20px rgba(255,79,145,.20);
}
.genre-chip div.stButton > button { min-height: 2.75rem; font-size: .88rem; }
.movie-card {
    background: linear-gradient(155deg, rgba(27,29,44,.98), rgba(16,17,27,.98));
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    padding: 1.15rem 1.1rem 1rem;
    min-height: 220px;
    box-shadow: 0 10px 26px rgba(0,0,0,.13);
    margin-bottom: .6rem;
}
.movie-icon { font-size: 1.65rem; margin-bottom: .45rem; }
.movie-title { font-family: 'Space Grotesk', sans-serif; color: #fff; font-size: 1.13rem; font-weight: 700; margin-bottom: .3rem; }
.movie-meta { color: #ff91ba; font-size: .78rem; font-weight: 700; letter-spacing: .03em; margin-bottom: .6rem; }
.movie-description { color: #bfc2d3; font-size: .87rem; line-height: 1.5; min-height: 3.9rem; }
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

# ---------- CINEMUSE WELCOME SOUND ----------
# Resolve the WAV beside app.py, even when Streamlit Cloud starts from another folder.
audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mixkit-relaxing-harp-sweep-2628.wav")
if os.path.isfile(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            welcome_audio = audio_file.read()
        st.audio(welcome_audio, format="audio/wav", autoplay=False)
    except OSError:
        st.warning("The welcome sound could not be loaded. Please check the WAV file in your repository.")
else:
    st.warning("Welcome sound file is missing. Upload mixkit-relaxing-harp-sweep-2628.wav beside app.py in GitHub.")

# ---------- ANIMATED AURORA / NETWORK BACKGROUND ----------
# This iframe runs the supplied HTML/CSS/JavaScript animation separately from Streamlit.
# The iframe is positioned behind the app using the CSS selector below.
st.markdown("""
<style>
/* Keep the animation component full-screen and behind all app content. */
div[data-testid="stElementContainer"]:has(iframe[title="streamlit_components.v1.html"]) {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 0 !important;
    pointer-events: none !important;
    margin: 0 !important;
    padding: 0 !important;
}
div[data-testid="stElementContainer"]:has(iframe[title="streamlit_components.v1.html"]) iframe {
    width: 100vw !important;
    height: 100vh !important;
    border: 0 !important;
    background: transparent !important;
    pointer-events: none !important;
}
/* Place Streamlit UI above the decorative background. */
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stMain"] {
    position: relative;
    z-index: 1;
}
[data-testid="stMain"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

import streamlit.components.v1 as components

components.html(r'''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { --bg:#070914; --c1:#6d5efc; --c2:#22d3ee; --c3:#f472b6; }
* { box-sizing:border-box; margin:0; }
html, body { width:100%; height:100%; background:transparent; color:#e8eaf6; font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif; overflow:hidden; }
.bg { position:fixed; inset:0; z-index:0; overflow:hidden; background:var(--bg); pointer-events:none; }
.blob { position:absolute; width:55vmax; height:55vmax; border-radius:50%; filter:blur(90px); opacity:.45; will-change:transform; animation:drift 24s ease-in-out infinite alternate; }
.b1 { background:var(--c1); top:-20vmax; left:-15vmax; }
.b2 { background:var(--c2); bottom:-25vmax; right:-15vmax; animation-duration:30s; opacity:.35; }
.b3 { background:var(--c3); top:30%; left:45%; width:35vmax; height:35vmax; animation-duration:36s; opacity:.22; }
@keyframes drift { from { transform:translate3d(0,0,0) scale(1); } to { transform:translate3d(8vmax,6vmax,0) scale(1.18); } }
.grid { position:absolute; inset:0; background-image:linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px); background-size:56px 56px; -webkit-mask-image:radial-gradient(ellipse at 50% 40%,#000 20%,transparent 75%); mask-image:radial-gradient(ellipse at 50% 40%,#000 20%,transparent 75%); }
#net { position:absolute; inset:0; width:100%; height:100%; }
.vignette { position:absolute; inset:0; background:radial-gradient(ellipse at center,transparent 40%,rgba(0,0,0,.48) 100%); }
.grain { position:absolute; inset:0; opacity:.06; mix-blend-mode:overlay; background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>"); }
@media (prefers-reduced-motion: reduce) { .blob { animation:none; } }
</style>
</head>
<body>
<div class="bg"><div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="grid"></div><canvas id="net"></canvas><div class="vignette"></div><div class="grain"></div></div>
<script>
(() => {
  const cv=document.getElementById('net'), ctx=cv.getContext('2d');
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W=0,H=0,DPR=1,nodes=[],mouse={x:-9999,y:-9999};
  const COLORS=['109,94,252','34,211,238','244,114,182'];
  function resize(){
    DPR=Math.min(devicePixelRatio||1,2); W=cv.clientWidth; H=cv.clientHeight;
    cv.width=W*DPR; cv.height=H*DPR; ctx.setTransform(DPR,0,0,DPR,0,0);
    const count=Math.round(Math.min(110,(W*H)/16000));
    nodes=Array.from({length:count},()=>({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:Math.random()*1.8+.8,c:COLORS[(Math.random()*COLORS.length)|0],p:Math.random()*Math.PI*2}));
  }
  function frame(t){
    ctx.clearRect(0,0,W,H); const LINK=Math.min(150,W/6);
    for(let i=0;i<nodes.length;i++){
      const a=nodes[i];
      if(!reduce){ a.x+=a.vx; a.y+=a.vy; if(a.x<0||a.x>W)a.vx*=-1; if(a.y<0||a.y>H)a.vy*=-1; const dx=a.x-mouse.x,dy=a.y-mouse.y,d=Math.hypot(dx,dy); if(d<140&&d>0){a.x+=dx/d*1.2;a.y+=dy/d*1.2;} }
      for(let j=i+1;j<nodes.length;j++){const b=nodes[j],d=Math.hypot(a.x-b.x,a.y-b.y);if(d<LINK){ctx.strokeStyle=`rgba(${a.c},${(1-d/LINK)*.35})`;ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}
      const pulse=.6+.4*Math.sin(t/900+a.p); const g=ctx.createRadialGradient(a.x,a.y,0,a.x,a.y,a.r*7);g.addColorStop(0,`rgba(${a.c},${.55*pulse})`);g.addColorStop(1,`rgba(${a.c},0)`);ctx.fillStyle=g;ctx.beginPath();ctx.arc(a.x,a.y,a.r*7,0,Math.PI*2);ctx.fill();ctx.fillStyle=`rgba(255,255,255,${.7*pulse+.3})`;ctx.beginPath();ctx.arc(a.x,a.y,a.r,0,Math.PI*2);ctx.fill();
    }
    if(!reduce)requestAnimationFrame(frame);
  }
  addEventListener('resize',resize);
  // Listen on the parent document too, so pointer response works while the iframe is behind the UI.
  try { parent.document.addEventListener('pointermove',e=>{mouse.x=e.clientX;mouse.y=e.clientY;}); } catch(e) {}
  addEventListener('pointermove',e=>{mouse.x=e.clientX;mouse.y=e.clientY;});
  addEventListener('pointerleave',()=>{mouse.x=mouse.y=-9999;});
  resize(); requestAnimationFrame(frame);
})();
</script>
</body>
</html>
''', height=1, scrolling=False)

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

# ---------- MOVIE TRAILERS ----------
# Verified YouTube trailer IDs are embedded directly in CineMuse.
# For titles without a verified ID, provide a YouTube search link instead of guessing a video.
trailer_ids = {
    "Inception": "ig1GrJI5dSg",
    "Interstellar": "zSWdZVtXT7E",
    "Pride and Prejudice": "1dYv5u6v55Y",
    "Toy Story": "KYz2wyBy3kc",
}
movies["trailer_id"] = movies["title"].map(trailer_ids)
movies["trailer_url"] = movies["title"].apply(
    lambda title: "https://www.youtube.com/results?search_query="
    + title.replace(" ", "+")
    + "+official+trailer"
)

def show_trailer(movie):
    """Display a trailer inline when a verified video ID is available."""
    with st.expander("▶ Watch trailer on this page", expanded=False):
        trailer_id = movie.get("trailer_id")
        if pd.notna(trailer_id) and trailer_id:
            st.video(f"https://www.youtube.com/watch?v={trailer_id}")
        else:
            st.info("An embedded trailer is not configured for this title yet.")
            st.link_button("Find official trailer on YouTube", movie["trailer_url"], use_container_width=True)
vectorizer = TfidfVectorizer(stop_words="english")
movie_vectors = vectorizer.fit_transform(movies["features"])
similarity = cosine_similarity(movie_vectors)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All"

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:12px;margin:.25rem 0 1.1rem;'>"
        "<div style='width:46px;height:46px;display:flex;align-items:center;justify-content:center;"
        "border-radius:15px;background:linear-gradient(135deg,#ff4f91,#9a6bff);"
        "box-shadow:0 8px 25px rgba(154,107,255,.22);font-size:1.45rem;'>🎬</div>"
        "<div><div style='font-family:Space Grotesk,sans-serif;font-size:1.55rem;font-weight:750;"
        "letter-spacing:-.05em;color:#f7f5ff;line-height:1.1;'>CineMuse</div>"
        "<div style='font-size:.68rem;letter-spacing:.14em;color:#969ab0;margin-top:5px;'>YOUR MOVIE UNIVERSE</div></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,rgba(255,79,145,.65),rgba(154,107,255,.12),transparent);margin:0 0 1.5rem;'></div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "EXPLORE",
        ["Home", "Discover", "Recommendations", "My Watchlist"],
        format_func=lambda x: {"Home": "⌂   Home", "Discover": "✦   Discover", "Recommendations": "✧   For You", "My Watchlist": "♡   My Watchlist"}[x],
        label_visibility="visible",
    )
    st.markdown(
        "<div style='height:1px;background:rgba(255,255,255,.08);margin:1.35rem 0 1.5rem;'></div>"
        "<div style='font-size:.72rem;letter-spacing:.14em;font-weight:800;color:#e7e2f5;margin-bottom:.85rem;'>A LITTLE MOVIE MAGIC</div>"
        "<div style='position:relative;overflow:hidden;padding:18px 16px;border-radius:19px;"
        "background:radial-gradient(circle at 100% 0%,rgba(255,79,145,.22),transparent 45%),"
        "linear-gradient(145deg,rgba(154,107,255,.14),rgba(255,255,255,.025));"
        "border:1px solid rgba(190,143,255,.18);'>"
        "<div style='font-size:1.3rem;margin-bottom:10px;'>✦ <span style='color:#ff8fbb;'>✧</span> ✦</div>"
        "<div style='font-family:Space Grotesk,sans-serif;font-size:1.08rem;font-weight:700;color:#fff;margin-bottom:6px;'>Find your next favourite.</div>"
        "<div style='font-size:.88rem;line-height:1.65;color:#b9b5cb;'>Little escapes. Big feelings. Stories made for your mood.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:1.2rem;font-size:.72rem;color:#777b91;'>MADE FOR YOUR NEXT MOVIE NIGHT · ✦</div>", unsafe_allow_html=True)

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
                show_trailer(movie)

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
            show_trailer(movie)

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
                show_trailer(movie)
                if st.button("Remove from watchlist", key=f"remove_{title}"):
                    st.session_state.watchlist.remove(title)
                    st.rerun()

st.markdown("<div style='text-align:center;color:#777b91;font-size:.78rem;padding:2rem 0 0;'>CINEMUSE &nbsp; ✦ &nbsp; DISCOVER SOMETHING THAT MOVES YOU</div>", unsafe_allow_html=True)
