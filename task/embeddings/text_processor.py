from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


SQL_QUERY = """
    SELECT text, 1 - (embedding {op} %(vector)s::vector) AS score
    FROM vectors
    WHERE 1 - (embedding {op} %(vector)s::vector) >= %(score_threshold)s
    ORDER BY score
    DESC LIMIT %(top_k)s
"""


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


SEARCH_MODE_TO_OPERATOR = {
    SearchMode.EUCLIDIAN_DISTANCE: '<->',
    SearchMode.COSINE_DISTANCE: '<=>'
}


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    #TODO:
    # provide method `process_text_file` that will:
    #   - apply file name, chunk size, overlap, dimensions and bool of the table should be truncated
    #   - truncate table with vectors if needed
    #   - load content from file and generate chunks (in `utils.text` present `chunk_text` that will help do that)
    #   - generate embeddings from chunks
    #   - save (insert) embeddings and chunks to DB
    #       hint 1: embeddings should be saved as string list
    #       hint 2: embeddings string list should be casted to vector ({embeddings}::vector)

    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, dimensions: int, truncate: bool):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            register_vector(conn)

            if truncate:
                cursor.execute('TRUNCATE vectors')

            with open(file_name) as fp:
                chunks = chunk_text(fp.read(), chunk_size, overlap)

                embeddings_data = self.embeddings_client.get_embeddings(chunks, dimensions)

                for idx, vector in embeddings_data.items():
                    cursor.execute('INSERT INTO vectors (document_name, text, embedding) values (%s, %s, %s::vector)', (file_name, chunks[idx], vector))

            conn.commit()

    #TODO:
    # provide method `search` that will:
    #   - apply search mode, user request, top k for search, min score threshold and dimensions
    #   - generate embeddings from user request
    #   - search in DB relevant context
    #     hint 1: to search it in DB you need to create just regular select query
    #     hint 2: Euclidean distance `<->`, Cosine distance `<=>`
    #     hint 3: You need to extract `text` from `vectors` table
    #     hint 4: You need to filter distance in WHERE clause
    #     hint 5: To get top k use `limit`

    def search(self, text: str, search_mode: SearchMode, top_k: int, score_threshold: float, dimensions: int) -> list[str]:
        embeddings_data = self.embeddings_client.get_embeddings(text, dimensions)
        vector = embeddings_data[0]

        sql_query = SQL_QUERY.format(op=SEARCH_MODE_TO_OPERATOR[search_mode])

        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query, {'vector': vector, 'score_threshold': score_threshold, 'top_k': top_k})
                return [row[0] for row in cursor.fetchall()]
