import streamlit as st
import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION

st.set_page_config(
    page_title="CinéRAG — Assistant Films",
    page_icon="🎬",
    layout="centered"
)

# STYLE MINIMALISTE MODERNE 

st.markdown("""
<style>
    /* Fond général */
    .stApp { background-color: #fafafa; }

    /* Titre */
    h1 { font-size: 1.6rem !important; font-weight: 500 !important; color: #000 !important; letter-spacing: 1px; }

    /* Masquer la barre supérieure Streamlit */
    header[data-testid="stHeader"] { background: transparent; }

    /* Chips de catégories */
    .chip {
        display: inline-block;
        background: #f0f0f0;
        color: #333;
        font-size: 12px;
        padding: 5px 12px;
        border-radius: 20px;
        margin: 3px;
        cursor: pointer;
    }

    /* Zone de saisie */
    .stChatInput input {
        background-color: #fff !important;
        border: 0.5px solid #ddd !important;
        border-radius: 8px !important;
        color: #000 !important;
        font-size: 14px !important;
    }

    /* Messages */
    [data-testid="stChatMessageContent"] {
        background-color: #f8f8f8 !important;
        border-radius: 10px !important;
        border: 0.5px solid #eee !important;
        color: #000 !important;
        font-size: 14px !important;
    }

    /* Expander sources */
    .streamlit-expanderHeader {
        background-color: #fff !important;
        border: 0.5px solid #eee !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        color: #333 !important;
    }
    .streamlit-expanderContent {
        background-color: #fafafa !important;
        border: 0.5px solid #eee !important;
    }

    /* Boutons suggestions */
    .stButton button {
        background-color: #f0f0f0 !important;
        color: #333 !important;
        border: none !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        padding: 4px 14px !important;
    }
    .stButton button:hover {
        background-color: #e0e0e0 !important;
    }

    /* Texte général */
    p, li, span { color: #333 !important; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# EN-TÊTE

col_logo, col_titre, col_info = st.columns([1, 6, 2])
with col_logo:
    st.markdown("<div style='background:#000; width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:18px; margin-top:6px;'>🎬</div>", unsafe_allow_html=True)
with col_titre:
    st.markdown("<h1 style='margin:0; padding-top:4px;'>CinéRAG</h1>", unsafe_allow_html=True)
with col_info:
    st.markdown("<p style='color:#999 !important; font-size:12px !important; text-align:right; margin-top:10px;'>1 000 films indexés</p>", unsafe_allow_html=True)

st.markdown("<hr style='border:none; border-top:0.5px solid #eee; margin:8px 0 16px;'>", unsafe_allow_html=True)

# BOUTONS SUGGESTIONS 

st.markdown("<p style='font-size:12px; color:#999 !important; margin-bottom:6px;'>Suggestions</p>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🤖 Intelligence artificielle"):
        st.session_state.suggestion = "Un film de science-fiction avec de l'intelligence artificielle ?"
with col2:
    if st.button("😱 Thriller psychologique"):
        st.session_state.suggestion = "Un thriller psychologique avec un retournement de situation ?"
with col3:
    if st.button("🎠 Animation familiale"):
        st.session_state.suggestion = "Un film d'animation familial pour toute la famille ?"
with col4:
    if st.button("💕 Romance"):
        st.session_state.suggestion = "Un film romantique émouvant ?"

st.markdown("<hr style='border:none; border-top:0.5px solid #eee; margin:16px 0;'>", unsafe_allow_html=True)

# CHARGEMENT

@st.cache_resource
def charger_ressources():
    index = faiss.read_index("index/films.index")
    with open("index/films_meta.json", "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)
    modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return index, chunks_avec_meta, modele, client

with st.spinner("Chargement de la base de films..."):
    index, chunks_avec_meta, modele, client = charger_ressources()

# FONCTIONS 

def rechercher(question, k=5):
    vecteur = modele.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(vecteur, k)
    resultats = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            resultats.append({
                "contenu": chunks_avec_meta[idx]["contenu"],
                "metadata": chunks_avec_meta[idx]["metadata"],
                "score": float(distances[0][i])
            })
    return resultats

def generer_reponse(question, chunks_pertinents):
    contexte = ""
    for i, chunk in enumerate(chunks_pertinents):
        contexte += f"\n--- Film {i+1} ---\n"
        contexte += chunk["contenu"]
        contexte += "\n"

    prompt_systeme = """Tu es un assistant expert en cinéma. Tu recommandes des films en te basant UNIQUEMENT
sur les fiches films fournies dans le contexte.

Règles strictes :
- Tu ne cites que des films présents dans le contexte fourni
- Pour chaque recommandation, tu mentionnes le titre, l'année et la note
- Si aucun film du contexte ne correspond à la demande, tu le dis clairement
- Tu argumentes chaque recommandation en 2-3 phrases
- Tu n'inventes jamais un film qui n'est pas dans le contexte"""

    prompt_utilisateur = f"""Voici les fiches de films disponibles :
{contexte}
Question : {question}
Réponds en te basant uniquement sur ces fiches."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_utilisateur}
        ]
    )
    return response.choices[0].message.content

# HISTORIQUE

if "historique" not in st.session_state:
    st.session_state.historique = []
if "suggestion" not in st.session_state:
    st.session_state.suggestion = None

for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.markdown(message["contenu"])

# SAISIE 

question_defaut = st.session_state.suggestion
if question_defaut:
    st.session_state.suggestion = None

question = st.chat_input("Décrivez le film que vous cherchez...")
question_finale = question or question_defaut

if question_finale:
    with st.chat_message("user"):
        st.markdown(question_finale)
    st.session_state.historique.append({"role": "user", "contenu": question_finale})

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            chunks = rechercher(question_finale, k=5)

        if chunks[0]["score"] > 200:
            st.warning("Aucun film très pertinent trouvé pour cette question.")

        with st.spinner("Rédaction des recommandations..."):
            reponse = generer_reponse(question_finale, chunks)

        st.markdown(reponse)

        with st.expander("🎞️ Films consultés pour cette réponse"):
            for c in chunks:
                m = c["metadata"]
                st.markdown(
                    f"<div style='padding:6px 0; border-bottom:0.5px solid #eee;'>"
                    f"<span style='font-weight:500; color:#000 !important;'>{m['titre']}</span> "
                    f"<span style='color:#999 !important;'>({m['annee']})</span> — "
                    f"<span style='color:#333 !important;'>⭐ {m['note']}/10</span> — "
                    f"<span style='color:#999 !important; font-size:12px;'>{m['genres']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.session_state.historique.append({"role": "assistant", "contenu": reponse})