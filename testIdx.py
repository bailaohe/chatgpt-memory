from typing import List, Dict, Any, Union

import redis
from redis.commands.search.field import TagField, VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from tqdm import tqdm

from chatgpt_memory.utils.openai_utils import openai_request

# define vector dimensions
VECTOR_DIMENSIONS = 1536
r = redis.Redis(host="localhost", password='foobared', port=6379)

INDEX_NAME = "index"  # Vector Index Name
DOC_PREFIX = "doc:"  # RediSearch Key Prefix for the Index

SAMPLE_QUERIES = ["Where is Berlin?"]
SAMPLE_DOCUMENTS = [
    {"text": "Berlin is located in Germany.", "conversation_id": "1"},
    {"text": "Vienna is in Austria.", "conversation_id": "1"},
    {"text": "Salzburg is in Austria.", "conversation_id": "2"},
]


def drop_index():
    # create the index
    try:
        r.ft(INDEX_NAME).dropindex(delete_documents=True)
    except:
        pass


def create_index(vector_dimensions: int, drop_existed=True, index_type='HNSW'):
    if drop_existed:
        drop_index()
    try:
        # check to see if index exists
        r.ft(INDEX_NAME).info()
        print("Index already exists!")
    except:
        if index_type == 'FLAT':
            # schema
            schema = (
                TagField("tag"),  # Tag Field Name
                VectorField("vector",  # Vector Field Name
                            "FLAT", {  # Vector Index Type: FLAT or HNSW
                                "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                                "DIM": vector_dimensions,  # Number of Vector Dimensions
                                "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                            }
                            ),
            )

            # index Definition
            definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)

            # create Index
            r.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
        else:
            r.ft(INDEX_NAME).create_index(
                [
                    VectorField(
                        'vector',
                        'HNSW',
                        {
                            "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                            "DIM": vector_dimensions,  # Number of Vector Dimensions
                            # "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                            # "TYPE": "FLOAT32",
                            "DISTANCE_METRIC": 'L2',
                            "INITIAL_CAP": 686,
                            "M": 40,
                            "EF_CONSTRUCTION": 200,
                        },
                    ),
                    TextField("text"),  # contains the original message
                    TagField("conversation_id"),  # `conversation_id` for each session
                ]
            )


def embed_azure(model: str, text: List[str]) -> np.ndarray:
    generated_embeddings: List[Any] = []

    headers: Dict[str, str] = {"Content-Type": "application/json", "api-key": 'd6bf81b1729340c0a460d6a589027ceb'}
    payload: Dict[str, Union[List[str], str]] = {"model": model, "input": text}

    res = openai_request(
        url="https://openai-tunnel.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15",
        headers=headers,
        payload=payload,
        timeout=30,
    )

    unordered_embeddings = [(ans["index"], ans["embedding"]) for ans in res["data"]]
    ordered_embeddings = sorted(unordered_embeddings, key=lambda x: x[0])

    generated_embeddings = [emb[1] for emb in ordered_embeddings]

    return np.array(generated_embeddings)


def embed_batch(model: str, text: List[str]) -> np.ndarray:
    all_embeddings = []
    for i in tqdm(
            range(0, len(text), 64),
            disable=False,
            desc="Calculating embeddings",
    ):
        batch = text[i: i + 64]
        batch_limited = batch
        generated_embeddings = embed_azure(model, batch_limited)
        all_embeddings.append(generated_embeddings)

    return np.concatenate(all_embeddings)


def embed_queries(queries: List[str]) -> np.ndarray:
    return embed_batch('text-embedding-ada-002', queries)


def embed_documents(docs: List[Dict]) -> np.ndarray:
    return embed_batch('text-embedding-ada-002', [d["text"] for d in docs])


create_index(vector_dimensions=VECTOR_DIMENSIONS)
document_embeddings: np.ndarray = embed_documents(SAMPLE_DOCUMENTS)
pipe = r.pipeline()
for idx, embedding in enumerate(document_embeddings):
    key = f"doc:{idx}"
    SAMPLE_DOCUMENTS[idx]["vector"] = embedding.astype(np.float32).tobytes()
    pipe.hset(key, mapping=SAMPLE_DOCUMENTS[idx])
# redis_datastore.index_documents(documents=SAMPLE_DOCUMENTS)
res = pipe.execute()

query_embeddings: np.ndarray = embed_queries(SAMPLE_QUERIES)
query_vector = query_embeddings[0].astype(np.float32).tobytes()
# search_results = redis_datastore.search_documents(query_vector=query_vector, conversation_id="1", topk=1)
# assert len(search_results), "No documents returned, expected 1 document."


query = (
    Query("*=>[KNN 2 @vector $vec as score]")
    .sort_by("score")
    .return_fields("id", "score")
    .paging(0, 2)
    .dialect(2)
)

query_params = {
    "vec": query_vector
}
print(r.ft(INDEX_NAME).search(query, query_params).docs)
