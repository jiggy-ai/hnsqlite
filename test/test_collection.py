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
            self.assertEqual(results[0].text, text)
            self.assertEqual(results[0].vector_as_array().tolist(), vector.tolist())
        # test reload collection from db produces same results
        newcollection = Collection.from_db(self.collection_name)
        for vector, text in zip(vectors, texts):
            results = newcollection.search(vector, k=1)
            self.assertEqual(results[0].text, text)
            self.assertEqual(results[0].vector_as_array().tolist(), vector.tolist())
        
        
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
        self.assertEqual(results[0].text, text)

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



class TestCollectionDelete(unittest.TestCase):

    def setUp(self):
        self.collection = Collection.create(name="test-collection", dim=2)
        vectors = [np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6])]
        texts = ["text1", "text2", "text3"]
        doc_ids = ["doc1", "doc2", "doc3"]
        metadata = [{"category": "A"}, {"category": "B"}, {"category": "A"}]
        self.collection.add_items(vectors, texts, doc_ids=doc_ids,  metadata=metadata)

    def tearDown(self):        
        for f in os.listdir("."):
            if f.endswith('.sqlite'):
                os.remove(f)
            if f.endswith('.hnsw'):
                os.remove(f)    
                
    def test_delete_all(self):
        self.collection.search(np.array([0.1, 0.2]), k=3)
        self.collection.delete(delete_all=True)
        with self.assertRaises(Exception):
            self.collection.search(np.array([0.1, 0.2]), k=3)

    def test_delete_with_filter(self):
        filter = {"category": "A"}
        self.collection.delete(filter=filter)
        results = self.collection.search(np.array([0.1, 0.2]), k=2)
        remaining_categories = [result.metadata["category"] for result in results]
        self.assertNotIn("A", remaining_categories)

    def test_delete_by_doc_id(self):
        doc_ids_to_delete = ["doc1", "doc3"]
        # first verify that the documents are there
        results = self.collection.search(np.array([0.1, 0.2]), k=3)        
        remaining_doc_ids = [result.doc_id for result in results]
        for doc_id in doc_ids_to_delete:
            self.assertIn(doc_id, remaining_doc_ids)
        # now delete them
        self.collection.delete(doc_ids=doc_ids_to_delete)
        # and verify that they are gone
        results = self.collection.search(np.array([0.1, 0.2]), k=2)
        remaining_doc_ids = [result.doc_id for result in results]
        for doc_id in doc_ids_to_delete:
            self.assertNotIn(doc_id, remaining_doc_ids)


class TestGetEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a test collection
        cls.collection = Collection.create(name="test-collection", dim=128)

        # Add some embeddings to the collection
        embeddings = [
            Embedding(vector=np.random.rand(128).tolist(), text="This is a test text 1."),
            Embedding(vector=np.random.rand(128).tolist(), text="This is a test text 2."),
            Embedding(vector=np.random.rand(128).tolist(), text="This is a test text 3."),
        ]
        cls.collection.add_embeddings(embeddings)

    @classmethod
    def tearDownClass(cls):
        # Delete the test collection
        cls.collection.delete(delete_all=True)
        for f in os.listdir("."):
            if f.endswith('.sqlite'):
                os.remove(f)
            if f.endswith('.hnsw'):
                os.remove(f)    
                
    def test_get_embeddings(self):
        # Test getting the first two embeddings
        embeddings = self.collection.get_embeddings(start=0, limit=2, reverse=False)
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0].text, "This is a test text 1.")
        self.assertEqual(embeddings[1].text, "This is a test text 2.")

        # Test getting the last embedding in reverse order
        embeddings = self.collection.get_embeddings(start=0, limit=1, reverse=True)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].text, "This is a test text 3.")


if __name__ == '__main__':
    unittest.main()
