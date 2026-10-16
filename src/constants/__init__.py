CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
K = 4
MAX_REWRITE_TRIES = 1
INDEX_NAME_CACHE_MEMORY = "crag-semantic-answer-cache"
INDEX_NAME = "self-corrective-rag-langgraph"
IMAGE_DIR = "vision_img/images"
DOCUMENTS = ["src/data/AI_Engineer-page1.pdf"]
CACHE_THRESHOLD = 0.90
CACHE_VERSION = "self-rag-v1"
CLEAR_CACHE = False
# FAISS returns squared L2 distances (lower is closer). A 0.30 cutoff was
# rejecting all retrieved chunks in normal document queries.
SIMILARITY_THRESHOLD_FIASS = 0.30
SIMILARITY_THRESHOLD_PINECONE = 0.30
FAISS_DB_PATH = "./faiss_index"

EXPECTED_DIMENSION = 1536      # must match your embedding model
EXPECTED_METRIC = "cosine"     # the graph treats Pinecone scores as similarity (higher = better)
CLOUD, REGION = "aws", "us-east-1"
READY_TIMEOUT_S = 90
