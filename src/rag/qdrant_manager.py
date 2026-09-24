import uuid
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models as rest
from src.config import config
from src.core.reranker import RerankerCandidate

logger = logging.getLogger(__name__)

VECTOR_SIZE = 1024  # bge-m3 default dimension

class QdrantManager:
    """Manages Qdrant collections, embeddings upserts, and vector retrieval."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or config.qdrant.host
        self.port = port or config.qdrant.port
        self.prefix = config.qdrant.collection_prefix
        self.client = AsyncQdrantClient(host=self.host, port=self.port)

    def _get_collection_name(self, subject_id: str) -> str:
        clean_id = "".join(c if c.isalnum() else "_" for c in subject_id.lower()).strip("_")
        return f"{self.prefix}{clean_id}"

    async def is_available(self) -> bool:
        """Check if Qdrant server is reachable."""
        try:
            collections = await self.client.get_collections()
            return True
        except Exception:
            return False

    async def ensure_subject_collection(self, subject_id: str):
        """Creates the subject collection if it does not already exist."""
        coll_name = self._get_collection_name(subject_id)
        try:
            existing = await self.client.get_collections()
            exists = any(c.name == coll_name for c in existing.collections)
            
            if not exists:
                logger.info(f"Creating Qdrant collection '{coll_name}' with dim={VECTOR_SIZE}...")
                await self.client.create_collection(
                    collection_name=coll_name,
                    vectors_config=rest.VectorParams(
                        size=VECTOR_SIZE,
                        distance=rest.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant to ensure collection '{coll_name}': {e}")

    async def upsert_chunks(
        self,
        subject_id: str,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
    ):
        """Upsert text chunks with corresponding bge-m3 embeddings."""
        await self.ensure_subject_collection(subject_id)
        coll_name = self._get_collection_name(subject_id)

        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            points.append(
                rest.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk
                )
            )

        if points:
            await self.client.upsert(
                collection_name=coll_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points into '{coll_name}'.")

    async def search(
        self,
        subject_id: str,
        query_vector: List[float],
        limit: int = 10,
    ) -> List[RerankerCandidate]:
        """Perform vector similarity search against subject collection."""
        coll_name = self._get_collection_name(subject_id)
        existing = await self.client.get_collections()
        exists = any(c.name == coll_name for c in existing.collections)
        if not exists:
            logger.warning(f"Collection '{coll_name}' does not exist.")
            return []

        search_result = await self.client.search(
            collection_name=coll_name,
            query_vector=query_vector,
            limit=limit,
        )

        candidates: List[RerankerCandidate] = []
        for scored_point in search_result:
            payload = scored_point.payload or {}
            text = payload.get("text", "")
            doc_name = payload.get("document_name", "Unknown Document")
            page_num = payload.get("page_number", "?")
            
            formatted_text = f"**[Course: {doc_name} | Page {page_num}]**\n{text}"
            candidates.append(
                RerankerCandidate(
                    text=formatted_text,
                    source="course_material",
                    metadata=payload,
                    score=float(scored_point.score)
                )
            )

        return candidates

    async def get_subject_stats(self, subject_id: str) -> Dict[str, Any]:
        """Get vector count and point stats for a subject."""
        coll_name = self._get_collection_name(subject_id)
        try:
            info = await self.client.get_collection(collection_name=coll_name)
            return {
                "collection_name": coll_name,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"collection_name": coll_name, "points_count": 0, "status": "not_created"}

    async def list_all_subjects(self) -> List[str]:
        """Returns list of registered subjects based on collections."""
        try:
            res = await self.client.get_collections()
            subjects = []
            for c in res.collections:
                if c.name.startswith(self.prefix):
                    subjects.append(c.name[len(self.prefix):])
            return subjects
        except Exception as e:
            logger.error(f"Failed to list Qdrant collections: {e}")
            return []
