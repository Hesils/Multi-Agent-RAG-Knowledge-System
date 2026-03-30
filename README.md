# 🔍 Multi-Agent RAG System

Un système de **Retrieval-Augmented Generation multi-agents** conçu pour interroger intelligemment une base documentaire hétérogène. Développé initialement pour explorer les architectures RAG avancées, il a été appliqué à un cas d'usage concret : permettre à une chercheuse de naviguer efficacement dans une bibliographie d'articles académiques.

---

## 🎯 Cas d'usage

> *"Comment retrouver rapidement une notion précise dans 50 articles académiques, et identifier les liens entre eux ?"*

Le système ingère des documents (PDF, texte, etc.), les indexe intelligemment, et expose un agent conversationnel capable de répondre à des questions complexes en s'appuyant sur les sources — avec une boucle de vérification pour limiter les hallucinations.

---

## 🏗️ Architecture

Le système est structuré en **3 pipelines indépendantes** :

```
📁 Dossier surveillé
       │
       ▼
┌─────────────────────┐
│  Ingestion Pipeline │  ← Watcher → Détection → Chunking → Contextualisation → ChromaDB
└─────────────────────┘

         Query utilisateur
               │
               ▼
┌─────────────────────┐
│   Answer Pipeline   │  ← Reformulation → Retrieval → Ranking → Answer/Critic loop
└─────────────────────┘

┌─────────────────────┐
│   Observabilité     │  ← Prometheus → Grafana + Trace Viewer (JS)
└─────────────────────┘
```

---

## ⚙️ Pipeline 1 — Ingestion automatisée

Un **watcher** surveille un dossier en temps réel et déclenche automatiquement l'ingestion, la mise à jour ou la suppression des documents dans la base vectorielle.

### Étapes clés

1. **Détection du type de fichier** — routing vers le loader adapté (PDF, texte, etc.)
2. **Chunking spécialisé** — découpage du contenu adapté au type de document via LangChain Community
3. **Contextualisation des chunks** — un agent reçoit chaque chunk *accompagné du contenu des pages adjacentes* (page n-1, n, n+1) pour produire un contexte enrichi, limitant la perte de sens aux frontières des chunks
4. **Persistance** — stockage des chunks contextualisés dans **ChromaDB**

> **Pourquoi contextualiser les chunks ?**  
> Un chunk isolé peut perdre son sens sans les éléments qui l'entourent. En passant le voisinage immédiat à l'agent de contextualisation, on préserve la cohérence sémantique et on améliore la qualité du retrieval.

---

## 🤖 Pipeline 2 — Answer Pipeline

### Reformulation de la requête

Avant le retrieval, un agent **reformule la question utilisateur** pour l'optimiser pour une recherche vectorielle (suppression du langage naturel conversationnel, extraction des concepts clés).

### Retrieval & Ranking

- Recherche dans **ChromaDB** (similarité vectorielle)
- **Re-ranking hybride** : combinaison avec **BM25** (similarité lexicale)
- Un **ChunkRankerAgent** filtre les chunks récupérés et ne conserve que ceux réellement pertinents pour la requête

### Boucle Answer / Critic

```
          ┌──────────────────────────────────┐
          │                                  │
          ▼                                  │  (si rejeté)
  AnswerAgent ──→ CriticAgent ──────────────►┘
                      │
                      │ (si approuvé)
                      ▼
               Réponse finale
```

- **AnswerAgent** : formule une réponse à partir des chunks sélectionnés
- **CriticAgent** : évalue la réponse selon plusieurs critères :
    - Présence des faits dans les chunks sources
    - Détection d'hallucinations
    - Pertinence par rapport à la question
- Si la réponse est rejetée, le **feedback est renvoyé à l'AnswerAgent** pour une nouvelle tentative

---

## 📊 Pipeline 3 — Observabilité

Le système expose des métriques et des outils de suivi pour monitorer le comportement des pipelines en production.

| Outil | Usage |
|---|---|
| **Prometheus** | Collecte de métriques (latence, taux d'appels, scores critic, ...) |
| **Grafana** | Dashboards de monitoring des pipelines |
| **Trace Viewer** (JS custom) | Suivi du cheminement complet de chaque requête individuelle |

---

## 🛠️ Stack technique

| Catégorie | Technologies                                            |
|---|---------------------------------------------------------|
| **Agents** | LangChain, OpenAI Agents SDK                            |
| **RAG / Vector Store** | ChromaDB                                                |
| **Document Loading** | LangChain Community loaders (multi-format)/Docling(pdf) |
| **Chunking** | LangChain Character Splitter                            |
| **Lexical Search** | BM25                                                    |
| **Observabilité** | Prometheus, Grafana, Trace Viewer (JS)                  |
| **Infrastructure** | Docker, Docker Compose                                  |

---

## 🚀 Lancement

```bash
# Cloner le projet
git clone https://github.com/Hesils/Multi-Agent-RAG-Knowledge-System
cd Multi-Agent-RAG-Knowledge-System
uv sync
# Lancer l'ensemble des services d'observabilité
cd Multi-Agent-RAG-Knowledge-System/rag_observability
docker compose up --build

# Lancer la pipeline d'ingestion automatisée
uv run main.py watch-datafiles-dir

# Lancer la Answer Pipeline
uv run main.py answer [query]
```

> ⚠️ Un fichier `.env` est nécessaire. Voir `.env.example` pour les variables requises (clé API OpenAI, configuration ChromaDB, etc.)


---

## 🔮 Améliorations envisagées

- [ ] Support de nouveaux types de documents (EPUB, HTML, Gdocs, ...)
- [ ] Interface utilisateur web pour l'agent conversationnel
- [ ] Évaluation automatisée de la qualité du RAG (RAGAS)
- [ ] Mémoire conversationnelle inter-sessions

---

## 👤 Auteur

Développé par Quentin Desvignes dans le cadre d'une montée en compétences sur les architectures LLM multi-agents.  
Profil LinkedIn : [linkedin.com/in/quentin-desvignes-8a8aa4139/](https://www.linkedin.com/in/quentin-desvignes-8a8aa4139/)