# 🎬 RAG Films — TP Python

Système de recommandation de films basé sur RAG (Retrieval-Augmented Generation).
Construit avec FAISS, sentence-transformers et l'API Groq, sans LangChain ni LlamaIndex.

## Fonctionnalités

- Recherche sémantique dans une base de 1000 films
- Recommandations argumentées générées par un LLM
- Citations des sources après chaque réponse
- Avertissement si aucun film pertinent n'est trouvé

## Installation

1. Cloner le dépôt
   git clone https://github.com/jane-monga/rag-films-tp.git
   cd rag-films-tp

2. Créer l'environnement virtuel
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux

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

## Architecture

PHASE 1 - INDEXATION (une seule fois)
CSV → Nettoyage → Conversion en texte → Chunking → Embedding → Index FAISS

PHASE 2 - INTERROGATION (à chaque question)
Question → Embedding → Recherche FAISS → Top 5 chunks → LLM Groq → Réponse

## Choix techniques

- Modèle embedding : paraphrase-multilingual-mpnet-base-v2 (multilingue, 768 dimensions)
- Chunking : taille max 800 caractères, overlap 100
- Base vectorielle : FAISS IndexFlatL2, sauvegardée sur disque
- LLM : llama-3.3-70b-versatile via Groq
- Temperature : 0.3 pour des réponses cohérentes

## Exemple de questions

- "Un film de science-fiction avec de l'intelligence artificielle ?"
- "Un thriller psychologique avec un retournement de situation ?"
- "Un film d'animation familial ?"