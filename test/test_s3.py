import os
import unittest
import numpy as np
from hnsqlite.collection import Collection, Embedding

# Set up the S3 configuration
BUCKET_NAME = 'jiggy-assets'
ENDPOINT_URL = os.environ.get("JIGGY_STORAGE_ENDPOINT_URL", "https://us-southeast-1.linodeobjects.com")
STORAGE_KEY_ID = os.environ['JIGGY_STORAGE_KEY_ID']
STORAGE_SECRET_KEY = os.environ['JIGGY_STORAGE_KEY_SECRET']


class TestS3BackupRestore(unittest.TestCase):

    def setUp(self):
        # Create a test collection
        self.collection_name = "test-s3-backup"
        self.dim = 100
        self.collection = Collection.create(name=self.collection_name, dim=self.dim)

        # Add 20 random vectors with text data and metadata
        self.embeddings = []
        for i in range(20):
            vector = np.random.rand(self.dim).astype(np.float32)
            text = f"Text {i}"
            doc_id = f"doc_{i}"
            metadata = {"key": f"value_{i}"}
            embedding = Embedding(vector=vector.tolist(), text=text, doc_id=doc_id, metadata=metadata)
            self.embeddings.append(embedding)
        self.collection.add_embeddings(self.embeddings)

    def test_backup_restore_s3(self):
        # Backup the collection to S3
        self.collection.backup_s3(bucket_name=BUCKET_NAME,
                                  endpoint_url=ENDPOINT_URL,
                                  storage_key_id=STORAGE_KEY_ID,
                                  storage_secret_key=STORAGE_SECRET_KEY)

        # Restore the collection from S3
        restored_collection = Collection.from_s3(collection_name=self.collection_name,
                                                 bucket_name=BUCKET_NAME,
                                                 endpoint_url=ENDPOINT_URL,
                                                 storage_key_id=STORAGE_KEY_ID,
                                                 storage_secret_key=STORAGE_SECRET_KEY)

        # Check if the restored collection has the same properties as the original collection
        self.assertEqual(self.collection.config.name, restored_collection.config.name)
        self.assertEqual(self.collection.config.dim, restored_collection.config.dim)

        # Verify that all of the data is unchanged after the restore
        for original_embedding in self.embeddings:
            search_results = restored_collection.search(original_embedding.vector_as_array(), k=1)
            restored_embedding = search_results[0]
            self.assertEqual(original_embedding.text, restored_embedding.text)
            self.assertEqual(original_embedding.doc_id, restored_embedding.doc_id)
            self.assertEqual(original_embedding.metadata, restored_embedding.metadata)

    def tearDown(self):
        # Clean up the test collection
        os.remove(f"collection_{self.collection_name}.sqlite")


if __name__ == '__main__':
    unittest.main()
