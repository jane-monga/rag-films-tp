import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# 1. CHARGEMENT DE L'INDEX 

print("⏳ Chargement de la base de connaissances...")
index = faiss.read_index("index/films.index")

with open("index/films_meta.json", "r", encoding="utf-8") as f:
    chunks_avec_meta = json.load(f)

modele = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(f"✅ Base chargée — {index.ntotal} films indexés\n")

# 2. FONCTION DE RECHERCHE 

def rechercher(question, k=5):
    """Trouve les k films les plus pertinents pour la question."""
    vecteur_question = modele.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(vecteur_question, k)

    resultats = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            resultats.append({
                "contenu": chunks_avec_meta[idx]["contenu"],
                "metadata": chunks_avec_meta[idx]["metadata"],
                "score": float(distances[0][i])
            })
    return resultats

# 3. PROMPT SYSTÈME 

def construire_prompt_systeme():
    return """Tu es un assistant expert en cinéma. Tu recommandes des films en te basant UNIQUEMENT
sur les fiches films fournies dans le contexte.

Règles strictes :
- Tu ne cites que des films présents dans le contexte fourni
- Pour chaque recommandation, tu mentionnes le titre, l'année et la note
- Si aucun film du contexte ne correspond à la demande, tu le dis clairement
- Tu argumentes chaque recommandation en 2-3 phrases
- Tu n'inventes jamais un film qui n'est pas dans le contexte"""

# 4. GÉNÉRATION DE LA RÉPONSE 

def generer_reponse(question, chunks_pertinents):
    """Génère une réponse en utilisant les chunks comme contexte."""
    contexte = ""
    for i, chunk in enumerate(chunks_pertinents):
        contexte += f"\n--- Film {i+1} ---\n"
        contexte += chunk["contenu"]
        contexte += "\n"

    prompt_utilisateur = f"""Voici les fiches de films disponibles :

{contexte}

Question de l'utilisateur : {question}

Réponds en te basant uniquement sur ces fiches."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            {"role": "user", "content": prompt_utilisateur}
        ]
    )
    return response.choices[0].message.content

# 5. BONUS : SCORE DE CONFIANCE 

def verifier_confiance(chunks):
    """Avertit si les résultats trouvés ne sont pas très pertinents."""
    if not chunks:
        return False
    meilleur_score = chunks[0]["score"]
    if meilleur_score > 200:
        print("⚠️  Avertissement : je n'ai pas trouvé de film très pertinent pour cette question.\n")
        return False
    return True

# 6. BOUCLE INTERACTIVE 
def main():
    print("🎬 Assistant Films RAG — tapez 'quit' pour quitter\n")

    while True:
        question = input("🎤 Votre question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        print("\n⏳ Recherche en cours...")
        chunks = rechercher(question, k=5)
        verifier_confiance(chunks)

        print("⏳ Génération de la réponse...\n")
        reponse = generer_reponse(question, chunks)

        print("─" * 60)
        print(reponse)
        print("─" * 60)

        print("\n📚 Sources consultées :")
        for c in chunks:
            m = c["metadata"]
            print(f"  • {m['titre']} ({m['annee']}) — {m['note']}/10")
        print()

if __name__ == "__main__":
    main()