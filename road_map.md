Answer pipeline
- Hybrid search
  - embeddings + BM25
- amélioration ranking
- métriques


Rag Pipeline
- Ingestion propre multi type
  - Implementer un CodeLoader
  - Implementer un GDocLoader
  - Loader pdf -> Marker ou Docling
- Une collection mais metadata enrichies
  - metadata = {
    "project": "projectA", ??
    "folder": "notes",
    "type": "pdf",
    "tags": [...] (Ai generated ?)
    }
- Préparation pour le fetch par AI Agents
  - chunk.metadata["keywords"] = extract_keywords(chunk) ???
  - générer des "relation chunks" — un agent qui, pour chaque nouvelle paire d'articles thématiquement proches, génère un micro-résumé des convergences/divergences et l'indexe comme un chunk synthétique.
- Smart chunking
  - PDF → par paragraphe
  - Code → par fonction
  - Markdown → par section
- Observabilité
  - filtres (query, latence, agent)
  - recherche texte
  - comparaison de traces