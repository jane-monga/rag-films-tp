import pandas as pd
import json

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
        return [texte]  # pas besoin de découper

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

# Test : afficher les infos de chunking
print(f"\n--- Infos chunking ---")
print(f"Documents : {len(documents)}")
print(f"Chunks : {len(chunks_avec_meta)}")
print(f"Différence : {len(chunks_avec_meta) - len(documents)} chunks supplémentaires (textes longs découpés)")