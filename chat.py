"""
RAG Chatbot - Version simplifiée utilisant TF-IDF au lieu de transformers
Plus léger et sans problème de dépendances PyTorch
"""

import subprocess
import sys
import os


def install_if_missing(package, pip_name=None):
    """Installe un package s'il manque"""
    if pip_name is None:
        pip_name = package
    try:
        __import__(package)
        return True
    except ImportError:
        print(f"Installation de {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])
        return True


# Installation des dépendances légères
print("🔍 Vérification des dépendances...")
packages = [
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("numpy", "numpy"),
    ("openai", "openai==0.28")
]

for module, pip_name in packages:
    install_if_missing(module, pip_name)

print("✅ Dépendances installées!\n")

# Imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict
import re
import openai

# ==================== CONFIGURATION ====================
DATA_PATH = "./suppliers_dataset.csv"
OPENAI_API_KEY = "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"
INDEX_PATH = "./tfidf_index.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 128


# ==================== FONCTIONS ====================

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def create_example_dataset(path: str = DATA_PATH):
    if os.path.exists(path):
        print(f"✓ Dataset existant: {path}")
        return

    print(f"📝 Création du dataset d'exemple...")
    data = {
        'supplier_id': ['SUP001', 'SUP002', 'SUP003', 'SUP004', 'SUP005'],
        'name': [
            'TechCorp Industries',
            'GlobalSupply Ltd',
            'FastDeliver SA',
            'QualityParts Inc',
            'ReliableGoods'
        ],
        'description': [
            'Fournisseur de composants électroniques basé en Asie',
            'Distributeur global de matières premières',
            'Service de logistique et livraison express',
            'Fabricant de pièces automobiles',
            'Grossiste en fournitures industrielles'
        ],
        'contracts': [
            'Contrat annuel 2024-2025, volume 500K€',
            'Accord cadre pluriannuel, 2M€ par an',
            'Contrat prestation services, 200K€',
            'Contrat exclusif pièces moteur, 800K€',
            'Contrat fournitures diverses, 150K€'
        ],
        'incidents': [
            'Retards de livraison en mars 2024 (5 jours), problème qualité lot #4521',
            'Aucun incident majeur signalé',
            'Retards récurrents (10 incidents en 6 mois), colis endommagés',
            'Non-conformité qualité sur 2% des pièces, rappel produit août 2024',
            'Factures incorrectes (3 cas), retards paiement'
        ],
        'financials': [
            'Santé financière moyenne, ratio dette 65%, cash-flow stable',
            'Excellente santé, notation AAA, croissance 15% annuelle',
            'Difficultés financières, pertes Q1-Q2 2024, restructuration en cours',
            'Bonne santé, investissements R&D importants',
            'Santé correcte, croissance modérée'
        ],
        'notes': [
            'Surveiller capacité production Q4. Audit qualité prévu novembre.',
            'Fournisseur stratégique fiable. Renouvellement contrat recommandé.',
            'RISQUE ÉLEVÉ: envisager fournisseur alternatif. Réunion urgente prévue.',
            'Bon partenaire, amélioration continue. Plan action qualité actif.',
            'Relation stable. Évaluation annuelle en décembre.'
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"✓ Dataset créé: {len(df)} fournisseurs\n")


def load_supplier_docs(path: str) -> List[Dict]:
    print(f"📂 Chargement du dataset: {path}")
    df = pd.read_csv(path)
    print(f"✓ {len(df)} fournisseurs chargés")

    docs = []
    columns = ['name', 'description', 'contracts', 'incidents', 'notes', 'financials']

    for idx, row in df.iterrows():
        text = " ".join(str(row.get(c, "")) for c in columns if c in df.columns)
        text = normalize_text(text)

        if not text:
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            docs.append({
                "supplier_id": row.get("supplier_id", f"row_{idx}"),
                "chunk_id": f"{row.get('supplier_id', idx)}_chunk{i}",
                "text": chunk
            })

    print(f"✓ {len(docs)} chunks créés\n")
    return docs


class SimpleRetriever:
    def __init__(self, documents=None, tfidf_matrix=None, vectorizer=None):
        self.documents = documents or []        # Toujours exister
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer

    def save(self, path):
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("ERROR: documents are empty — cannot save index")

        if self.tfidf_matrix is None:
            raise ValueError("ERROR: tfidf_matrix is None — index was not built before saving")

        index_data = {
            "documents": self.documents,
            "tfidf_matrix": self.tfidf_matrix.toarray().tolist(),
            "vectorizer": self.vectorizer.__dict__
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        retriever = SimpleRetriever()

        retriever.documents = data.get("documents", [])
        retriever.tfidf_matrix = np.array(data["tfidf_matrix"])
        retriever.vectorizer = TfidfVectorizer()
        retriever.vectorizer.__dict__.update(data["vectorizer"])

        return retriever

    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_idx = scores.argsort()[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in top_idx]



def build_prompt(query: str, contexts: List[Dict]) -> str:
    ctx_texts = "\n---\n".join([
        f"[{c['supplier_id']} | score={c['score']:.3f}]\n{c['text']}"
        for c in contexts
    ])

    return f"""Tu es un expert en gestion des risques fournisseurs. 
Utilise le contexte suivant pour répondre à la question. Si tu n'es pas certain, dis-le.

CONTEXTE:
{ctx_texts}

QUESTION: {query}

RÉPONSE (résumé concis + indicateurs de risque + citations):"""


def call_openai(prompt: str, api_key: str) -> str:
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Erreur API OpenAI: {str(e)}\n\nContexte récupéré:\n{prompt[:500]}..."


def rag_query(query: str, retriever: SimpleRetriever, api_key: str, top_k: int = 5):
    print(f"🔍 Recherche: '{query}'")

    contexts = retriever.retrieve(query, top_k=top_k)
    print(f"✓ {len(contexts)} documents trouvés")

    prompt = build_prompt(query, contexts)
    answer = call_openai(prompt, api_key)

    return {"answer": answer, "contexts": contexts}


# ==================== PROGRAMME PRINCIPAL ====================

def main():
    print("=" * 70)
    print("🤖 RAG CHATBOT - VERSION SIMPLIFIÉE (TF-IDF)")
    print("=" * 70)
    print()

    # 1. Dataset
    create_example_dataset(DATA_PATH)
    docs = load_supplier_docs(DATA_PATH)

    # 2. Index
    if os.path.exists(INDEX_PATH):
        print(f"✓ Index existant trouvé: {INDEX_PATH}")
        retriever = SimpleRetriever.load(INDEX_PATH)
    else:
        retriever = SimpleRetriever(docs)
        retriever.save(INDEX_PATH)

    # 3. Requêtes
    print("=" * 70)
    print("📋 EXEMPLES DE REQUÊTES")
    print("=" * 70)
    print()

    queries = [
        "Quels fournisseurs ont des retards de livraison?",
        "Y a-t-il des problèmes financiers?",
        "Incidents qualité récents?"
    ]

    for i, q in enumerate(queries, 1):
        print("─" * 70)
        print(f"REQUÊTE {i}/{len(queries)}")
        print("─" * 70)

        result = rag_query(q, retriever, OPENAI_API_KEY, top_k=5)

        print(f"\n💬 RÉPONSE:")
        print(result['answer'])

        print(f"\n📎 SOURCES:")
        for ctx in result['contexts'][:3]:
            print(f"  • {ctx['supplier_id']} (score: {ctx['score']:.3f})")
        print()

    print("=" * 70)
    print("✅ TERMINÉ")
    print("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()