from loguru import logger
import unittest
import numpy as np
from hnsqlite.collection import Collection, Embedding
import os

class TestSearchWithFilter(unittest.TestCase):

    def setUp(self):
        logger.info("setUp")
        self.collection_name = "test-collection"
        self.dim = 2
        self.collection = Collection.create(self.collection_name, self.dim)
        logger.info(f"collection: {self.collection}")
        
        # Add some test embeddings with metadata
        embeddings = [
            Embedding(vector=[0.1, 0.2], text="item1", metadata={"category": "A", "value": 1}),
            Embedding(vector=[0.3, 0.4], text="item2", metadata={"category": "B", "value": 2}),
            Embedding(vector=[0.5, 0.6], text="item3", metadata={"category": "A", "value": 3}),
            Embedding(vector=[0.7, 0.8], text="item4", metadata={"category": "B", "value": 4})
        ]
        logger.warning(embeddings)
        self.collection.add_embeddings(embeddings)


    def tearDown(self):
        os.remove(f"collection_test-collection.sqlite")
        for f in os.listdir("."):
            if f.endswith('.hnsw'):
                os.remove(f)
        
    def test_search_filter_category(self):
        query_vector = np.array([0.1, 0.2])
        filter_dict = {"category": {"$eq": "A"}}
        results = self.collection.search(query_vector, k=2, filter=filter_dict)

        # Check if the results only contain items with category "A"
        for result in results:
            self.assertEqual(result.item.metadata["category"], "A")

    def test_search_filter_value(self):
        query_vector = np.array([0.1, 0.2])
        filter_dict = {"value": {"$gt": 2}}
        results = self.collection.search(query_vector, k=2, filter=filter_dict)

        # Check if the results only contain items with value > 2
        for result in results:
            self.assertGreater(result.item.metadata["value"], 2)

    def test_search_filter_and(self):
        query_vector = np.array([0.1, 0.2])
        filter_dict = {"$and": [{"category": {"$eq": "A"}}, {"value": {"$gt": 1}}]}
        results = self.collection.search(query_vector, k=2, filter=filter_dict)

        # Check if the results only contain items with category "A" and value > 1
        for result in results:
            self.assertEqual(result.item.metadata["category"], "A")
            self.assertGreater(result.item.metadata["value"], 1)

    def test_search_filter_or(self):
        query_vector = np.array([0.1, 0.2])
        filter_dict = {"$or": [{"category": {"$eq": "A"}}, {"value": {"$gt": 3}}]}
        results = self.collection.search(query_vector, k=2, filter=filter_dict)

        # Check if the results only contain items with category "A" or value > 3
        for result in results:
            self.assertTrue(result.item.metadata["category"] == "A" or result.item.metadata["value"] > 3)

if __name__ == "__main__":
    unittest.main()
