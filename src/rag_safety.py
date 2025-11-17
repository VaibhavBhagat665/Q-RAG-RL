import numpy as np
from typing import List, Dict, Any
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _ST_AVAILABLE = False
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import chromadb  # type: ignore
    _CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    _CHROMA_AVAILABLE = False

class RAGSafetyModule:
    def __init__(self):
        self._use_st = False
        self.model = None
        self._tfidf: TfidfVectorizer | None = None
        if _ST_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._use_st = True
            except Exception:
                self.model = None
                self._use_st = False
        if not self._use_st:
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2048)
        self._use_chroma = _CHROMA_AVAILABLE
        self.collection = None
        self.docs: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        if self._use_chroma:
            try:
                self.client = chromadb.Client()
                try:
                    self.collection = self.client.get_collection("safety_constraints")
                except Exception:
                    self.collection = self.client.create_collection("safety_constraints")
            except Exception:
                self._use_chroma = False
                self.collection = None
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self):
        safety_rules = [
            {
                "constraint": "Battery SOC must be between 10% and 90%",
                "category": "battery",
                "threshold_min": 0.1,
                "threshold_max": 0.9,
                "penalty": 1000
            },
            {
                "constraint": "Bus voltage must be between 0.95 and 1.05 pu",
                "category": "voltage",
                "threshold_min": 0.95,
                "threshold_max": 1.05,
                "penalty": 500
            },
            {
                "constraint": "Battery charge rate cannot exceed 1 MW",
                "category": "power",
                "threshold_max": 1.0,
                "penalty": 800
            },
            {
                "constraint": "Battery discharge rate cannot exceed 1 MW",
                "category": "power",
                "threshold_max": -1.0,
                "penalty": 800
            },
            {
                "constraint": "Grid frequency must be between 49.5 and 50.5 Hz",
                "category": "frequency",
                "threshold_min": 49.5,
                "threshold_max": 50.5,
                "penalty": 1200
            },
            {
                "constraint": "Total generation must not exceed 110% of load",
                "category": "balance",
                "threshold_max": 1.1,
                "penalty": 600
            }
        ]
        
        documents = [rule["constraint"] for rule in safety_rules]
        if self._use_st and self.model is not None:
            embeddings = self.model.encode(documents)
        else:
            assert self._tfidf is not None
            self._tfidf.fit(documents)
            embeddings = self._tfidf.transform(documents).toarray()
        if self._use_chroma and self.collection is not None:
            if self.collection.count() == 0:
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=safety_rules,
                    ids=[f"rule_{i}" for i in range(len(safety_rules))]
                )
        else:
            self.docs = documents
            self.metas = safety_rules
            self.embeddings = np.array(embeddings, dtype=float)
    
    def check_safety(self, action: float, current_state: Dict[str, Any]) -> tuple[bool, float]:
        violations = []
        total_penalty = 0
        
        battery_soc = current_state.get('battery_soc', 0.5)
        voltage = current_state.get('voltage', 1.0)
        frequency = current_state.get('frequency', 50.0)
        total_gen = current_state.get('total_generation', 0)
        load = current_state.get('load', 1.0)
        
        if battery_soc < 0.1 or battery_soc > 0.9:
            violations.append("Battery SOC violation")
            total_penalty += 1000
            
        if abs(action) > 1.0:
            violations.append("Battery power limit violation")
            total_penalty += 800
            
        if voltage < 0.95 or voltage > 1.05:
            violations.append("Voltage violation")
            total_penalty += 500
            
        if frequency < 49.5 or frequency > 50.5:
            violations.append("Frequency violation")
            total_penalty += 1200
            
        if total_gen > 1.1 * load:
            violations.append("Generation-load balance violation")
            total_penalty += 600
            
        is_safe = len(violations) == 0
        return is_safe, total_penalty
    
    def get_relevant_constraints(self, query: str, n_results: int = 3):
        if self._use_st and self.model is not None:
            query_embedding = self.model.encode([query])
            qe = np.array(query_embedding[0], dtype=float)
        else:
            if self._tfidf is None or self.embeddings is None:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            qe = self._tfidf.transform([query]).toarray()[0]
        if self._use_chroma and self.collection is not None:
            results = self.collection.query(
                query_embeddings=[qe.tolist()],
                n_results=n_results
            )
            return results
        if self.embeddings is None or len(self.docs) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        emb = self.embeddings
        qe_norm = qe / (np.linalg.norm(qe) + 1e-9)
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = emb_norm @ qe_norm
        idx = np.argsort(-sims)[:n_results]
        return {
            "ids": [[f"mem_{i}" for i in idx.tolist()]],
            "documents": [[self.docs[i] for i in idx.tolist()]],
            "metadatas": [[self.metas[i] for i in idx.tolist()]],
            "distances": [[float(1 - sims[i]) for i in idx.tolist()]]
        }
    
    def add_constraint(self, constraint_text: str, metadata: Dict[str, Any]):
        embedding = self.model.encode([constraint_text])
        if self._use_chroma and self.collection is not None:
            next_id = f"custom_{self.collection.count()}"
            self.collection.add(
                embeddings=embedding.tolist(),
                documents=[constraint_text],
                metadatas=[metadata],
                ids=[next_id]
            )
        else:
            if self.embeddings is None:
                self.embeddings = np.array(embedding, dtype=float)
            else:
                self.embeddings = np.vstack([self.embeddings, np.array(embedding, dtype=float)])
            self.docs.append(constraint_text)
            self.metas.append(metadata)
