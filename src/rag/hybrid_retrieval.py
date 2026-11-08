import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _tokenize(text):
    return [token.casefold() for token in re.findall(r"\w+|[^\w\s]", str(text or ""), flags=re.UNICODE)]


def reciprocal_rank_fusion(rankings, k=60):
    if k <= 0:
        raise ValueError("k must be positive")
    scores = defaultdict(float)
    sources = defaultdict(set)
    for source_index, ranking in enumerate(rankings):
        source_name = "ranker_{}".format(source_index + 1)
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item if isinstance(item, str) else item["chunk_id"]
            scores[chunk_id] += 1.0 / (k + rank)
            sources[chunk_id].add(source_name)
    return [
        {"chunk_id": chunk_id, "rrf_score": score, "retrieval_sources": sorted(sources[chunk_id])}
        for chunk_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


class HybridRetriever:
    def __init__(self, chunks, vector_encoder=None, rrf_constant=60, include_security_flagged=False):
        candidate_chunks = []
        seen_ids = set()
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("Every indexed chunk must be a JSON object")
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            chunk_text = chunk.get("chunk_text")
            if not chunk_id or not isinstance(chunk_text, str) or not chunk_text.strip():
                raise ValueError("Every indexed chunk requires a unique chunk_id and non-empty chunk_text")
            if chunk_id in seen_ids:
                raise ValueError("Duplicate chunk_id in retrieval index: {}".format(chunk_id))
            seen_ids.add(chunk_id)
            candidate_chunks.append(dict(chunk))
        self.security_flagged_chunk_count = sum(bool(chunk.get("security_review_required")) for chunk in candidate_chunks)
        self.chunks = [
            chunk for chunk in candidate_chunks
            if include_security_flagged or not chunk.get("security_review_required")
        ]
        self.vector_encoder = vector_encoder
        self.rrf_constant = rrf_constant
        self._chunk_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self._tokenized = [_tokenize(chunk["chunk_text"]) for chunk in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._vectorizer = None
        self._tfidf_matrix = None
        self._dense_vectors = None
        if self.chunks and vector_encoder is None:
            self._vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=200000)
            self._tfidf_matrix = self._vectorizer.fit_transform([chunk["chunk_text"] for chunk in self.chunks])
        elif self.chunks:
            self._dense_vectors = np.asarray(vector_encoder.encode([chunk["chunk_text"] for chunk in self.chunks]), dtype=float)
        self.vector_model = getattr(vector_encoder, "model_id", "tfidf-character-ngram-v1")

    def _filtered_indices(self, filters):
        if not filters:
            return list(range(len(self.chunks)))
        selected = []
        for index, chunk in enumerate(self.chunks):
            matched = True
            for field, expected in filters.items():
                actual = str(chunk.get(field, "")).casefold()
                options = expected if isinstance(expected, (list, tuple, set)) else [expected]
                if actual not in {str(option).casefold() for option in options}:
                    matched = False
                    break
            if matched:
                selected.append(index)
        return selected

    def _rank_keyword(self, query, candidate_indices):
        if not candidate_indices or self._bm25 is None:
            return [], {}
        all_scores = self._bm25.get_scores(_tokenize(query))
        ordered = sorted(candidate_indices, key=lambda index: (-float(all_scores[index]), self.chunks[index]["chunk_id"]))
        return ordered, {index: float(all_scores[index]) for index in candidate_indices}

    def _rank_vector(self, query, candidate_indices):
        if not candidate_indices:
            return [], {}
        if self.vector_encoder is None:
            query_vector = self._vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self._tfidf_matrix[candidate_indices]).ravel()
        else:
            query_vector = np.asarray(self.vector_encoder.encode([query]), dtype=float)
            candidate_vectors = self._dense_vectors[candidate_indices]
            numerator = candidate_vectors.dot(query_vector[0])
            denominators = np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query_vector[0])
            scores = np.divide(numerator, denominators, out=np.zeros_like(numerator), where=denominators != 0)
        score_map = {index: float(score) for index, score in zip(candidate_indices, scores)}
        ordered = sorted(candidate_indices, key=lambda index: (-score_map[index], self.chunks[index]["chunk_id"]))
        return ordered, score_map

    def retrieve(self, query, top_k=5, filters=None, mode="hybrid"):
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError("mode must be bm25, vector, or hybrid")
        candidate_indices = self._filtered_indices(filters)
        if not candidate_indices or not str(query).strip():
            return []
        keyword_order, keyword_scores = self._rank_keyword(query, candidate_indices)
        vector_order, vector_scores = self._rank_vector(query, candidate_indices)
        selected_rankings = {
            "bm25": [keyword_order],
            "vector": [vector_order],
            "hybrid": [keyword_order, vector_order],
        }[mode]
        fused = reciprocal_rank_fusion([
            [self.chunks[index]["chunk_id"] for index in ranking]
            for ranking in selected_rankings
        ], self.rrf_constant)
        results = []
        for rank_record in fused[:top_k]:
            chunk_id = rank_record["chunk_id"]
            chunk = dict(self._chunk_by_id[chunk_id])
            source_index = next(index for index in candidate_indices if self.chunks[index]["chunk_id"] == chunk_id)
            chunk.update({
                "bm25_score": keyword_scores.get(source_index, 0.0),
                "vector_score": vector_scores.get(source_index, 0.0),
                "rrf_score": rank_record["rrf_score"] if mode == "hybrid" else None,
                "retrieval_mode": mode,
                "vector_model": self.vector_model,
            })
            results.append(chunk)
        return results
