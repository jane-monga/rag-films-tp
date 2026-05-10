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
Sur 4800 films indexés, 4953 chunks ont été créés au total.
153 films avaient des synopsis suffisamment longs pour être
découpés en plusieurs chunks.
La taille de 800 caractères avec un overlap de 100 a été choisie
pour préserver le contexte aux jointures sans perdre d'information.

### Modèle d'embedding
Choix de paraphrase-multilingual-mpnet-base-v2 car il gère
le français et l'anglais. Utile car les synopsis sont en anglais
mais les questions peuvent être posées en français.

### Base vectorielle
FAISS IndexFlatL2 sauvegardé sur disque en deux fichiers :
- index/films.index : les vecteurs (4953 vecteurs de dimension 768)
- index/films_meta.json : les métadonnées associées
Cela évite de recalculer les vecteurs à chaque lancement.
L'indexation complète sur 4800 films a pris environ 22 minutes.

### Prompt système
Temperature à 0.3 pour rester fidèle au contexte tout en
gardant une formulation fluide. Le prompt interdit explicitement
au LLM d'inventer des films absents du contexte.

### Bonus implémenté — Score de confiance
Score de confiance : si le meilleur score FAISS dépasse 200,
un avertissement s'affiche. Les scores observés lors des tests
étaient entre 4 et 10, ce qui indique une très bonne pertinence.

### Interface Streamlit
Ajout d'une interface web avec Streamlit pour rendre le RAG
accessible sans ligne de commande. L'interface inclut :
- 4 boutons de suggestions cliquables pour guider l'utilisateur
- Un historique de conversation persistant pendant la session
- Un affichage des sources consultées par réponse (titre, année, note, genres)
- Un style minimaliste moderne adapté au thème cinéma

Le choix de Streamlit s'est imposé pour sa simplicité : quelques
lignes de Python suffisent pour obtenir une vraie interface web,
sans avoir à écrire du HTML ou du JavaScript.

### Évolution du corpus
Le corpus a été étendu de 1000 à 4800 films en cours de développement
pour améliorer la qualité des recommandations et couvrir un plus
large spectre de genres et d'époques.