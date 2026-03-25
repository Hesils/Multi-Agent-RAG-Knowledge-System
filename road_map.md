Answer pipeline
- Hybrid search
  - embeddings + BM25
- amélioration ranking
- métriques


Rag Pipeline
- Ingestion propre multi type
  - Implementer un CodeLoader
  - Implementer un GDocLoader
- Une collection mais metadata enrichies
  - metadata = {
    "project": "projectA", ??
    "folder": "notes",
    "type": "pdf",
    "tags": [...] (Ai generated ?)
    }
- Préparation pour le fetch par AI Agents
  - chunk.metadata["keywords"] = extract_keywords(chunk) ???
- Smart chunking
  - PDF → par paragraphe
  - Code → par fonction
  - Markdown → par section
- Observabilité
  - logs d’ingestion
  - nombre de chunks
  - erreurs
  - fichiers ignorés