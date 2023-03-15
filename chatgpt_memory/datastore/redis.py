from typing import Any, Dict, List
from uuid import uuid4
import redis
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.query import Query

from chatgpt_memory.datastore.config import RedisDataStoreConfig
from chatgpt_memory.datastore.datastore import DataStore


class RedisDataStore(DataStore):
    def __init__(self, config: RedisDataStoreConfig, do_flush_data: bool = False):
        super().__init__(config=config)
        self.config = config

        self.do_flush_data = do_flush_data

    def connect(self):
        """
        Connect to the Redis server.
        """
        connection_pool = redis.ConnectionPool(
            host=self.config.host, port=self.config.port, password=self.config.password
        )
        self.redis_connection = redis.Redis(connection_pool=connection_pool)

        # flush data only once after establishing connection
        if self.do_flush_data:
            self.redis_connection.flushall()
            self.do_flush_data = False

    def create_index(self):
        """
        Creates a Redis index with a dense vector field.
        """
        self.redis_connection.ft().create_index(
            [
                VectorField(
                    self.config.vector_field_name,
                    self.config.index_type,
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.config.vector_dimensions,
                        "DISTANCE_METRIC": self.config.distance_metric,
                        "INITIAL_CAP": self.config.number_of_vectors,
                        "M": self.config.M,
                        "EF_CONSTRUCTION": self.config.EF,
                    },
                ),
                TextField("text"),  # contains the original message
                TagField("conversation_id"),  # `conversation_id` for each session
            ]
        )

    def index_documents(self, documents: List[Dict]):
        """
        Indexes the set of documents.

        Args:
            documents (List[Dict]): List of documents to be indexed.
        """
        redis_pipeline = self.redis_connection.pipeline(transaction=False)
        for document in documents:
            assert (
                "text" in document and "conversation_id" in document
            ), "Document must include the fields `text`, and `conversation_id`"
            redis_pipeline.hset(uuid4().hex, mapping=document)
        redis_pipeline.execute()

    def search_documents(
        self,
        query_vector: bytes,
        conversation_id: str,
        topk: int = 5,
    ) -> List[Any]:
        """
        Searches the redis index using the query vector.

        Args:
            query_vector (np.ndarray): Embedded query vector.
            topk (int, optional): Number of results. Defaults to 5.
            result_fields (int, optional): Name of the fields that you want to be
                                           returned from the search result documents

        Returns:
            List[Any]: Search result documents.
        """
        query = (
            Query(
                f"""(@conversation_id:{{{conversation_id}}})=>[KNN {topk} \
                    @{self.config.vector_field_name} $vec_param AS vector_score]"""
            )
            .sort_by("vector_score")
            .paging(0, topk)
            .return_fields(
                # parse `result_fields` as strings separated by comma to pass as params
                "convsersation_id",
                "vector_score",
                "text",
            )
            .dialect(2)
        )
        params_dict = {"vec_param": query_vector}
        result_documents = self.redis_connection.ft().search(query, query_params=params_dict).docs

        return result_documents