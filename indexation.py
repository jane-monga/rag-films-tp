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

# Test : afficher le premier document pour vérifier
print("\n--- Exemple de document ---")
print(documents[0]["contenu"])
print("Métadonnées :", documents[0]["metadata"])