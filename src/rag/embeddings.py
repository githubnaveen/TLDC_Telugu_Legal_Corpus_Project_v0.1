from pathlib import Path
import argparse
import json

from src.rag.chunking import write_jsonl


class LocalSentenceTransformerEncoder:
    """Optional encoder that loads an already-downloaded model directory only."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError("Provide a local, licence-reviewed embedding model directory")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("Install optional sentence-transformers to use dense embeddings") from error
        self.model = SentenceTransformer(str(self.model_path), model_kwargs={"local_files_only": True})
        self.model_id = str(self.model_path)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def embed_chunks(chunks, encoder, include_security_flagged=False):
    chunks = [chunk for chunk in chunks if include_security_flagged or not chunk.get("security_review_required")]
    if not chunks:
        return []
    vectors = encoder.encode([chunk["chunk_text"] for chunk in chunks])
    embedded = []
    for chunk, vector in zip(chunks, vectors):
        record = dict(chunk)
        record["embedding"] = [float(value) for value in vector]
        record["embedding_model"] = str(encoder.model_id)
        record["embedding_status"] = "READY"
        embedded.append(record)
    return embedded


def main():
    parser = argparse.ArgumentParser(description="Embed prepared TLDC chunks using a local-only Sentence Transformers model.")
    parser.add_argument("--chunks", default="data/embedded_chunks/tldc_embedded_chunks.jsonl")
    parser.add_argument("--model", required=True, help="Local model directory; remote model IDs are not accepted")
    parser.add_argument("--output", default="data/embedded_chunks/tldc_embedded_chunks.jsonl")
    args = parser.parse_args()
    chunk_path = Path(args.chunks)
    with chunk_path.open(encoding="utf-8") as source:
        chunks = [json.loads(line) for line in source if line.strip()]
    if not chunks:
        print("No approved chunks to embed; no model was loaded.")
        write_jsonl(args.output, [])
        return
    encoder = LocalSentenceTransformerEncoder(args.model)
    write_jsonl(args.output, embed_chunks(chunks, encoder))
    print("Embedded chunks:", len(chunks), "Model:", encoder.model_id)


if __name__ == "__main__":
    main()
