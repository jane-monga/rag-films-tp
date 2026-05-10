import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# CHARGEMENT DE L'INDEX 

print("⏳ Chargement de la base de connaissances...")
index = faiss.read_index("index/films.index")

with open("index/films_meta.json", "r", encoding="utf-8") as f:
    chunks_avec_meta = json.load(f)

modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
print(f"✅ {index.ntotal} films indexés\n")

# TEST DE RECHERCHE VECTORIELLE 

question = "film de science fiction avec intelligence artificielle"
print(f"Question test : {question}\n")

vecteur = modele.encode([question], convert_to_numpy=True).astype(np.float32)
distances, indices = index.search(vecteur, 3)

print("--- Résultats de la recherche ---")
for i, idx in enumerate(indices[0]):
    meta = chunks_avec_meta[idx]["metadata"]
    print(f"Résultat {i+1} : {meta['titre']} ({meta['annee']}) — {meta['note']}/10")
    print(f"  Score FAISS : {distances[0][i]:.2f}")
    print()