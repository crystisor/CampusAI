import os
from pathlib import Path
from typing import Optional, Literal
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class DiscordSettings(BaseModel):
    token: str = Field(default_factory=lambda: os.getenv("DISCORD_BOT_TOKEN", ""))
    status_message: str = "Studying College Subjects"

class OllamaSettings(BaseModel):
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_model: str = "Spark-X2.5-4b-Q8_0"
    llm_keep_alive: str = "-1"  # Permanently keep LLM hot in GPU VRAM
    think: bool = False         # Explicitly disable model thinking/drafts to output only final answer
    llm_temperature: float = 0.2
    llm_num_ctx: int = 4096
    llm_top_p: float = 0.9
    llm_top_k: int = 40
    llm_repeat_penalty: float = 1.05
    router_model: str = "Arch-Router"
    router_num_gpu: int = 0     # Offload to CPU
    router_keep_alive: str = "30m"
    router_temperature: float = 0.0
    router_max_tokens: int = 10
    embedding_model: str = "bge-m3"
    embedding_num_gpu: int = 0  # Offload to CPU
    embedding_keep_alive: str = "30m"
    request_timeout: float = 120.0

class RerankerSettings(BaseModel):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    max_length: int = 768
    initial_qdrant_candidates: int = 15
    initial_web_candidates: int = 8
    score_threshold: float = 0.0  # Minimum raw reranker score, inclusive
    device: str = "auto"

class EmbeddingSettings(BaseModel):
    vector_size: int = 1024
    distance: Literal["cosine"] = "cosine"
    batch_size: int = Field(default=10, gt=0)

class QdrantSettings(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    grpc_port: int = 6334
    prefer_grpc: bool = False
    collection_prefix: str = "subject_"

class SearchSettings(BaseModel):
    provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_results: int = 8

class IngestionSettings(BaseModel):
    layout_model: str = "pp-doclayoutV3"
    ocr_model: str = "glm-ocr"
    storage_dir: Path = BASE_DIR / "storage" / "subjects"
    dpi: int = 200
    chunk_size: int = 900
    chunk_overlap: int = 150
    math_density_threshold: float = 0.35

class WebSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

class AppConfig(BaseModel):
    discord: DiscordSettings = Field(default_factory=DiscordSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    web: WebSettings = Field(default_factory=WebSettings)

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"
    
    config_dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                config_dict = loaded

    # Handle string env var replacements in yaml
    def resolve_env_strings(obj):
        if isinstance(obj, dict):
            return {k: resolve_env_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_env_strings(x) for x in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.getenv(var_name, "")
        return obj

    resolved = resolve_env_strings(config_dict)
    return AppConfig(**resolved)

# Global configuration singleton
config = load_config()
