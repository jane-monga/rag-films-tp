import pandas as pd
import json

# Chargement des données
df = pd.read_csv("data/tmdb_5000_movies.csv")
print(f"✅ {len(df)} films chargés")
print(df.columns.tolist())
print(df.head(2))