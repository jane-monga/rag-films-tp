import pandas as pd
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer

# ─── 1. CHARGER LES DONNÉES ───────────────────────────────────────────────────

df = pd.read_csv("data/tmdb_5000_movies.csv")
print(f"✅ {len(df)} films chargés")

# ─── 2. NETTOYER ET CONVERTIR EN TEXTE ───────────────────────────────────────

def extraire_genres(genres_json):
    """Transforme '[{"id": 18, "name": "Drama"}]' en 'Drama, Romance'"""
    try:
        genres = json.loads(genres_json)
        return ", ".join([g["name"] for g in genres])
    except:
        return "Inconnu"

def film_vers_texte(row):
    """Convertit une ligne du CSV en texte riche à embedder."""
    titre = row.get("title", "Inconnu")
    synopsis = row.get("overview", "")
    genres = extraire_genres(row.get("genres", "[]"))
    annee = str(row.get("release_date", ""))[:4]
    note = row.get("vote_average", 0)
    langue = row.get("original_language", "")

    texte = f"Film : {titre} ({annee})\n"
    texte += f"Genres : {genres}\n"
    texte += f"Note : {note}/10\n"
    texte += f"Langue originale : {langue}\n"
    texte += f"Synopsis : {synopsis}"
    return texte

# Garder seulement les films avec un synopsis
df = df[df["overview"].notna() & (df["overview"] != "")].copy()
df = df.head(1000)
print(f"✅ {len(df)} films après nettoyage")

# Construire la liste de documents
documents = []
for _, row in df.iterrows():
    documents.append({
        "contenu": film_vers_texte(row),
        "metadata": {
            "titre": row.get("title", "Inconnu"),
            "annee": str(row.get("release_date", ""))[:4],
            "note": row.get("vote_average", 0),
            "genres": extraire_genres(row.get("genres", "[]")),
            "langue": row.get("original_language", ""),
        }
    })

print(f"✅ {len(documents)} documents préparés")

# ─── 3. CHUNKING ──────────────────────────────────────────────────────────────

def chunker(texte, taille_max=800, overlap=100):
    """
    Découpe un texte long en chunks avec chevauchement.
    - taille_max : nombre max de caractères par chunk
    - overlap : caractères répétés entre deux chunks consécutifs
    """
    if len(texte) <= taille_max:
        return [texte]

    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille_max
        chunk = texte[debut:fin]
        chunks.append(chunk)
        debut += taille_max - overlap
    return chunks

# Appliquer le chunking sur tous les documents
chunks_avec_meta = []
for doc in documents:
    morceaux = chunker(doc["contenu"])
    for morceau in morceaux:
        chunks_avec_meta.append({
            "contenu": morceau,
            "metadata": doc["metadata"]
        })

print(f"✅ {len(chunks_avec_meta)} chunks créés")

# ─── 4. EMBEDDINGS ────────────────────────────────────────────────────────────

print("⏳ Chargement du modèle d'embedding...")
modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

print("⏳ Calcul des vecteurs (quelques minutes)...")
textes = [c["contenu"] for c in chunks_avec_meta]
vecteurs = modele.encode(textes, show_progress_bar=True, convert_to_numpy=True)
vecteurs = vecteurs.astype(np.float32)

print(f"✅ Vecteurs calculés — forme : {vecteurs.shape}")

# ─── 5. CRÉATION ET SAUVEGARDE DE L'INDEX FAISS ──────────────────────────────

dimension = vecteurs.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vecteurs)
print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs")

# Sauvegarder sur disque
os.makedirs("index", exist_ok=True)
faiss.write_index(index, "index/films.index")

with open("index/films_meta.json", "w", encoding="utf-8") as f:
    json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)

print("✅ Index sauvegardé dans index/")
print("🎉 Indexation terminée ! Tu peux maintenant lancer rag.py")