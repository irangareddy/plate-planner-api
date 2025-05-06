# 🥗 Plate Planner

**Plate Planner** is a graph-powered, ML-enhanced meal assistant. It uses FastAPI, Neo4j, and pre-trained embeddings to recommend recipes and provide smart ingredient substitutions — all backed by a structured graph and semantic similarity models.

---

## 🚀 Features

* 🍽️ **Recipe Suggestion** using sentence embeddings + ingredient overlap
* 🔁 **Ingredient Substitution** with context-aware and similarity-based edges
* 🧠 **Neo4j Graph Bootstrap** — Ingredients, Recipes, Relationships
* 🧰 **Fully Dockerized**: FastAPI + Neo4j + bootstrap tasks
* 📦 Uses `poetry` for Python dependency management
* 🧪 Production-ready `Taskfile.yml` for repeatable dev commands

---

## 🧱 Architecture Overview

```
          +--------------+               +------------------+
          |  FastAPI     |  <—— REST ——> |   Neo4j (Graph)   |
          |   (Uvicorn)  |               |   Ingredient ↔ Recipe
          +--------------+               |   SIMILAR_TO edges
                 ↑                       |   SUBSTITUTES_WITH
                 |                       +------------------+
                 |
          [ML Model Inference]
          [SentenceTransformer + FAISS]
```

---

## 📦 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/plate-planner-api.git
cd plate-planner-api
```

### 2. Build and run (Docker)

```bash
docker-compose up --build
```

> ✅ This:
>
> * Starts Neo4j
> * Bootstraps graph from CSV + models
> * Starts FastAPI on `http://localhost:8000`

---

## 🔢 Available Endpoints

Once up:

* **Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Neo4j Browser**: [http://localhost:7474](http://localhost:7474)

  * User: `neo4j`, Password: `12345678`

---

## 🧪 Tasks for Dev

Run from root (requires `task` CLI):

```bash
task install           # poetry install
task serve             # Run FastAPI with uvicorn
task neo4j:bootstrap   # Load graph: ingredients, recipes, embeddings
```

---

## 📁 Project Structure

```
src/
├── api/             # FastAPI app + routes
├── config/          # Paths, secrets, Neo4j URI
├── data/            # Raw, processed, model, result files
├── database/        # Graph load logic (ingredients, edges)
├── evaluation/      # Graph stats, ranx, analysis
├── utils/           # Model inference, NER, vector helpers
```

---

## 🧠 Technologies

* Python 3.11
* FastAPI + Uvicorn
* Neo4j (Bolt + HTTP)
* FAISS + SentenceTransformers
* Poetry + Taskfile
* Docker Compose

---

## ⚠️ Notes

* Data is persisted via Docker volumes.
* Use `task neo4j:reset` (optional) to wipe and reload.
* Avoid hardcoded paths — everything resolves via `config.paths.DataPaths`.

---