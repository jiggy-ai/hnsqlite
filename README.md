# hnsqlite

`hnsqlite` is a text-centric integration of SQLite and [HNSWLIB](https://github.com/nmslib/hnswlib) to provide a persistent collection of embeddings (strings, vectors, and metadata) and search time filtering based on the metadata.

## Classes

### Collection

The `Collection` class represents a combination of a SQLite database and an HNSWLIB index. The purpose of this class is to provide a persistent collection of embeddings (strings, vectors, and metadata) and search time filtering based on the metadata.

#### Methods

- `Collection()`:- Initializes a new Collection as a SQLite database file and associated HNSWLIB index. If the specified collection name is found in the database, the collection will be initialized from database. Otherwise, a new collection of the specified name will be created in the database.
- `save_index`: Saves the current index after updates.
- `make_index`: Creates an HNSW index that includes all embeddings in the collection database and uses this new index for the collection going forward.
- `load_index`: Loads the latest HNSW index from disk and uses it for the collection.
- `add_items`: Adds new items to the collection.
- `add_embedding`: Adds a single Embedding object to the collection.
- `add_embeddings`: Adds a list of Embedding objects to the collection.
- `get_embeddings`: return a list of embeddings from an offset
- `get_embeddings_doc_ids`: return a list of embeddings associated with specified doc_ids
- `search`: Queries the HNSW index for the nearest neighbors of the given vector. Supply a k parameter (defaults to 12) and an optional filter dictionary.
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
collection = Collection(collection_name="example", dim=128)

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

# Filtering


The filtering function is designed to support metadata filtering similar to MongoDB.  It utilizes the hnswlib filtering function to accept or reject nearest neighbot candidates based on the embedding metadata matching a search time filtering criteria.


## Supported Metadata

The embedding `metadata` is a dictionary that stores metadata associated with items in the collection. The keys represent the field names of the metadata, and the supported values are strings, numbers, booleans or lists of strings.

Example of a metadata dictionary:

```python
{
    "author": "John Doe",
    "rating": 4.5,
    "tags": ["python", "database", "search"]
}
```

## Filtering Operations

The  search function supports a filter similar to [MongoDB](https://www.mongodb.com/docs/manual/reference/operator/query/). 

The following operations are supported:

- `$eq`: Checks if a metadata value is equal to the specified value.
- `$ne`: Checks if a metadata value is not equal to the specified value.
- `$gt`: Checks if a metadata value is greater than the specified value.
- `$gte`: Checks if a metadata value is greater than or equal to the specified value.
- `$lt`: Checks if a metadata value is less than the specified value.
- `$lte`: Checks if a metadata value is less than or equal to the specified value.
- `$in`: Checks if a metadata value is in the specified list of values.
- `$nin`: Checks if a metadata value is not in the specified list of values.
- `$and`: Combines multiple filter conditions using an AND logical operator.
- `$or`: Combines multiple filter conditions using an OR logical operator.

## Usage

```python
filter_dict = {
    "rating": {"$gte": 4},
    "tags": {"$in": ["python", "search"]},
    "$or": [
        {"author": {"$eq": "John Doe"}},
        {"author": {"$eq": "Jane Smith"}}
    ]
}

metadata_dict = {
    "author": "John Doe",
    "rating": 4.5,
    "tags": ["python", "database", "search"]
}

result = filter_item(filter_dict, metadata_dict)
```

The `result` will be `True` if the `metadata_dict` satisfies the conditions defined in the `filter_dict`. In this example, the metadata has a rating greater than or equal to 4 and at least one tag from the specified list, the author is either "John Doe" or "Jane Smith", so the result will be `True`.`

This will create a new collection with 10 random embeddings, get the number of items in the collection, search for the 5 nearest neighbors of a random query vector.
