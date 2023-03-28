# hnsqlite

`hnsqlite` is a text-centric integration of SQLite and HNSWLIB to provide a persistent collection of embeddings (strings, vectors, and metadata) and search time filtering based on the metadata.

## Classes

### Collection

The `Collection` class represents a combination of a SQLite database and an HNSWLIB index. The purpose of this class is to provide a persistent collection of embeddings (strings, vectors, and metadata) and search time filtering based on the metadata.

#### Methods

- `create`: Initializes a new Collection as a SQLite database file and associated HNSWLIB index.
- `from_db`: Creates a Collection object from a SQLite collection database file.
- `save_index`: Saves the current index after updates.
- `make_index`: Creates an HNSW index that includes all embeddings in the collection database and uses this new index for the collection going forward.
- `load_index`: Loads the latest HNSW index from disk and uses it for the collection.
- `add_items`: Adds new items to the collection.
- `add_embedding`: Adds a single Embedding object to the collection.
- `add_embeddings`: Adds a list of Embedding objects to the collection.
- `search`: Queries the HNSW index for the nearest neighbors of the given vector.
- `delete`: Deletes items from the collection based on a filter, a specific list of document_ids, or everything.

### dbHnswIndexConfig

The `dbHnswIndexConfig` class represents the configuration associated with an HNSWLIB index as stored in the database.

### dbCollectionConfig

The `dbCollectionConfig` class represents the configuration associated with a collection of strings and embeddings as persisted in the database.

### dbEmbedding

The `dbEmbedding` class represents an embedding as stored in the database.

### Embedding

The `Embedding` class represents an Embedding as sent to/from the Collection API.

### SearchResponse

The `SearchResponse` class represents the response of a search operation, containing the item (embedding) and its distance from the query vector.


## Usage

To use `hnsqlite`, you can create a new collection, add items to it, and perform search operations. Here's an example:


```python
from hnsqlite import Collection
import numpy as np

# Create a new collection
collection = Collection.create(name="example", dim=128)

# Add items to the collection
vectors = [np.random.rand(128) for _ in range(10)]
texts = [f"Text {i}" for i in range(10)]
collection.add_items(vectors, texts)

# Get the number of items in the collection
item_count = collection.count()
print(f"Number of items in the collection: {item_count}")

# Search for the nearest neighbors of a query vector
query_vector = np.random.rand(128)
results = collection.search(query_vector, k=5)

# Print the search results
for result in results:
    print(f"Item: {result}, Distance: {result.distance}")

```

This will create a new collection with 10 random embeddings, get the number of items in the collection, search for the 5 nearest neighbors of a random query vector.
