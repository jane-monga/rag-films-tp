# Compte-rendu — TP RAG Films

## Difficultés rencontrées

- Le modèle llama3-8b-8192 était déprécié au moment du TP.
  Solution : remplacé par llama-3.3-70b-versatile après consultation
  de la documentation Groq.

- Le fichier .env n'était pas lu correctement au départ.
  Solution : activation du paramètre python.terminal.useEnvFile
  dans les settings VS Code.

- Le téléchargement du modèle d'embedding (1.1 Go) a pris
  plusieurs minutes à la première exécution.

- Conflit Git lors du premier push car GitHub avait généré
  un fichier LICENSE automatiquement.
  Solution : git pull --allow-unrelated-histories puis push.

## Décisions de conception

### Chunking
Les fiches films sont courtes (moins de 800 caractères en moyenne).
La plupart ne sont donc pas découpées. Sur 1000 films, seulement
29 ont généré plusieurs chunks, soit 1029 chunks au total.
La taille de 800 caractères avec un overlap de 100 a été choisie
pour préserver le contexte aux jointures.

### Modèle d'embedding
Choix de paraphrase-multilingual-mpnet-base-v2 car il gère
le français et l'anglais. Utile car les synopsis sont en anglais
mais les questions peuvent être posées en français.

### Base vectorielle
FAISS IndexFlatL2 sauvegardé sur disque en deux fichiers :
- index/films.index : les vecteurs
- index/films_meta.json : les métadonnées associées
Cela évite de recalculer les vecteurs à chaque lancement.

### Prompt système
Temperature à 0.3 pour rester fidèle au contexte tout en
gardant une formulation fluide. Le prompt interdit explicitement
au LLM d'inventer des films absents du contexte.

### Bonus implémenté
Score de confiance : si le meilleur score FAISS dépasse 200,
un avertissement s'affiche. Les scores observés lors des tests
étaient entre 4 et 10, ce qui indique une très bonne pertinence.