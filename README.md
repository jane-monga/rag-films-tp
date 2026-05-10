# 🎬 RAG Films — TP Python

Système de recommandation de films basé sur RAG (Retrieval-Augmented Generation).
Construit avec FAISS, sentence-transformers et l'API Groq.

## Installation

1. Cloner le dépôt
   git clone https://github.com/TON_PSEUDO/rag-films-tp.git
   cd rag-films-tp

2. Créer l'environnement virtuel
   python -m venv venv
   venv\Scripts\activate  # Windows

3. Installer les dépendances
   pip install -r requirements.txt

4. Créer un fichier .env avec ta clé Groq
   GROQ_API_KEY=ta_clé_ici

5. Télécharger les données
   - Aller sur kaggle.com/datasets/tmdb/tmdb-movie-metadata
   - Télécharger tmdb_5000_movies.csv
   - Le placer dans le dossier data/

6. Lancer l'indexation (une seule fois)
   python indexation.py

7. Lancer le RAG
   python rag.py

## Choix techniques

- **Modèle embedding** : paraphrase-multilingual-mpnet-base-v2 (multilingue, 768 dimensions)
- **Chunking** : taille max 800 caractères, overlap 100
- **Base vectorielle** : FAISS IndexFlatL2, sauvegardée sur disque
- **LLM** : llama-3.3-70b-versatile via Groq
- **Temperature** : 0.3 pour des réponses cohérentes

## Architecture

PHASE 1 - INDEXATION (une seule fois)
CSV → Nettoyage → Conversion en texte → Chunking → Embedding → Index FAISS

PHASE 2 - INTERROGATION (à chaque question)
Question → Embedding → Recherche FAISS → Top 5 chunks → LLM Groq → Réponse