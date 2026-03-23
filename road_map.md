Answer pipeline
- Hybrid search
  - embeddings + BM25
- amélioration ranking
- métriques


Rag Pipeline
- Détection des changements (watcher)
  - Ajout du bind event -> action
- Gestion intelligente des updates
  - Détection du file modified
  - Chunking du file
  - recuperation des chunks en VDB de ce fichier
  - comparaison de chunck généré et des chunks en VDB (chunk hash) pour savoir lequels delet
  - stocké tous ceux qui n'ont pas trouvé leur équivalent en VDB
- Ingestion propre multi type
  - Faire une fonction get_loader en fonction de l'extension du fichier
  - Implementer un TextLoader (.txt, .md)
  - Implementer un CodeLoader
  - Implement un OpenDocLoader (.odt)
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