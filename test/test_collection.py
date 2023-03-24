from loguru import logger
import unittest
from hnsqlite import Collection, Embedding
import numpy as np
import os


class TestCollection(unittest.TestCase):

    def setUp(self):
        logger.info("setUp")
        self.collection_name = "test-collection"
        self.dim = 5
        self.collection = Collection.create(self.collection_name, self.dim)
        logger.info(f"collection: {self.collection}")

    def tearDown(self):
        logger.info("tearDown")
        os.remove(f"collection_{self.collection_name}.sqlite")
        for f in os.listdir("."):
            if f.endswith('.hnsw'):
                os.remove(f)
        

    def test_create_collection(self):
        logger.info("test_create_collection")
        self.assertIsNotNone(self.collection)
        self.assertEqual(self.collection.config.name, self.collection_name)
        self.assertEqual(self.collection.config.dim, self.dim)

    def test_add_items(self):
        logger.info("test_add_items")
        vectors = [np.random.rand(self.dim).astype(np.float32) for _ in range(3)]
        texts = ["text1", "text2", "text3"]
        self.collection.add_items(vectors, texts)
        for vector, text in zip(vectors, texts):
            results = self.collection.search(vector, k=1)
            self.assertEqual(results[0].item.text, text)
            self.assertEqual(results[0].item.vector_as_array().tolist(), vector.tolist())
        # test reload collection from db produces same results
        newcollection = Collection.from_db(self.collection_name)
        for vector, text in zip(vectors, texts):
            results = newcollection.search(vector, k=1)
            self.assertEqual(results[0].item.text, text)
            self.assertEqual(results[0].item.vector_as_array().tolist(), vector.tolist())
        
        
class TestCollection2(unittest.TestCase):
    def tearDown(self):        
        for f in os.listdir("."):
            if f.endswith('.sqlite'):
                os.remove(f)
            if f.endswith('.hnsw'):
                os.remove(f)    
                
    def test_create_collection_valid(self):
        logger.info("test_create_collection_valid")
        collection = Collection.create("test-collection1", 128)
        self.assertIsNotNone(collection)

    def test_create_collection_invalid(self):
        logger.info("test_create_collection_invalid")
        with self.assertRaises(Exception):
            Collection.create("", 128)
        with self.assertRaises(Exception):
            Collection.create("test_collection", -128)

    def test_load_collection_existing(self):
        logger.info("test_load_collection_existing")
        collection = Collection.create("test-collection2", 128)
        loaded_collection = Collection.from_db("test-collection2")
        self.assertIsNotNone(loaded_collection)

    def test_load_collection_nonexistent(self):
        logger.info("test_load_collection_nonexistent")
        with self.assertRaises(Exception):
            Collection.from_db("nonexistent-collection")

    def test_add_single_item(self):
        logger.info("test_add_single_item")
        collection = Collection.create("test-collection3", 128)
        vector = np.random.rand(128).astype(np.float32)
        text = "Test text"
        embedding = Embedding(vector=vector, text=text)
        collection.add_embedding(embedding)
        results = collection.search(vector, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].item.text, text)

    def test_add_multiple_items(self):
        logger.info("test_add_multiple_items")
        collection = Collection.create("test-collection4", 128)
        vectors = [np.random.rand(128).astype(np.float32) for _ in range(5)]
        texts = [f"Test text {i}" for i in range(5)]
        collection.add_items(vectors, texts)
        results = collection.search(vectors[0], k=5)
        self.assertEqual(len(results), 5)

    def test_add_items_invalid(self):
        logger.info("test_add_items_invalid")
        collection = Collection.create("test-collection5", 128)
        vectors = [np.random.rand(128).astype(np.float32) for _ in range(5)]
        texts = [f"Test text {i}" for i in range(4)]  # One less text than vectors
        with self.assertRaises(Exception):
            collection.add_items(vectors, texts)

    def test_search_valid(self):
        logger.info("test_search_valid")
        collection = Collection.create("test-collection6", 128)
        vector = np.random.rand(128).astype(np.float32)
        collection.add_items([vector], ["Test text"])
        results = collection.search(vector, 1)
        self.assertEqual(len(results), 1)

    def test_search_invalid(self):
        logger.info("test_search_invalid")
        collection = Collection.create("test-collection7", 128)
        vector = np.random.rand(64).astype(np.float32)  # Wrong length
        with self.assertRaises(Exception):
            collection.search(vector)

if __name__ == '__main__':
    unittest.main()
