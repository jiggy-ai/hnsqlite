# hnsqlite

`hnsqlite` is a text-centric integration of SQLite and [Hnswlib](https://github.com/nmslib/hnswlib) to provide a persistent collection of embeddings (strings, vectors, and metadata) and search time filtering based on the metadata.

## Classes

### Collection

The `Collection` class represents a combination of a SQLite database and an HNSWLIB index. The purpose of this class is to provide a persistent collection of embeddings (strings, vectors, and metadata) and search time filtering based on the metadata.

### Embedding

`Embedding` is a class that represents an embedding sent to or received from the Collection API.

**Attributes:**

- `vector`: A list of float values representing the user-supplied vector element. The vector can be sent as a numpy array and will be converted to a list of floats.
- `text`: The text that was input to the model to generate this embedding.
- `doc_id`: An optional document_id associated with the embedding.
- `metadata`: An optional dictionary of metadata associated with the text.
- `created_at`: The epoch timestamp when the embedding was created (in seconds).

### SearchResponse

`SearchResponse` is a class derived from the `Embedding` class, specifically designed for returning search results. A `SearchResponse` object consists of an embedding along with its distance to the query vector.

**Attributes:**

- `vector`: A list of float values representing the user-supplied vector element. The vector can be sent as a numpy array and will be converted to a list of floats.
- `text`: The text that was input to the model to generate this embedding.
- `doc_id`: An optional document_id associated with the embedding.
- `metadata`: An optional dictionary of metadata associated with the text.
- `created_at`: The epoch timestamp when the embedding was created (in seconds).
- `distance`: A float value representing the cosine similarity distance between the search result and the query vector.  Lower distances represent closer matches.


#### Collection Methods

- `Collection()`:- Initializes a new Collection as a SQLite database file and associated HNSWLIB index. If the specified collection name is found in the database, the collection will be initialized from database. Otherwise, a new collection of the specified name will be created in the database.
- `add_items`: Adds new items to the collection as lists of individual components.  An alternative interface to add_embeddings().
- `add_embedding`: Adds a single Embedding object to the collection.  A convenience alternative to add_embeddings().
- `add_embeddings`: Adds a list of Embedding objects to the collection.  An alternative interface to add_items().
- `get_embeddings`: return a list of embeddings from an offset
- `get_embeddings_doc_ids`: return a list of embeddings associated with specified doc_ids
- `search`: Queries the HNSW index for the nearest neighbors of the given vector. Supply a k parameter (defaults to 12) and an optional filter dictionary.
- `delete`: Deletes items from the collection based on a filter, a specific list of document_ids, or everything.


### Database classes

The following classes are the internal SqlModel data classes used to persist the embeddings and configuration in sqlite. They are not directly accessed by the user, but will be created as tables in the sqlite database:
 - The `dbHnswIndexConfig` class represents the configuration associated with an HNSWLIB index as stored in the database.  
 - The `dbCollectionConfig` class represents the configuration associated with a collection of strings and embeddings as persisted in the database.
 - The `dbEmbedding` class represents an embedding as stored in the database.



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
