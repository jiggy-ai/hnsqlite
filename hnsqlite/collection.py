"""
hnsqlite
A text-centric integration of sqlite and hnswlib to provide a persistent collection of embeddings (strings+vectors+metadata) 
and search time filtering based on the metadata.
"""

from loguru import logger
from sqlmodel import SQLModel, create_engine, Session, select, Field, Column, LargeBinary,  desc, asc
from pydantic import EmailStr, BaseModel, ValidationError, validator, root_validator
import hnswlib
from typing import List, Optional, Tuple, Union
from time import time
from psutil import cpu_count
import os
from json import dumps, loads
import numpy as np
from .filter import filter_item
from .util import md5_file, _is_valid_namestr
import sqlite3

CPU_COUNT = cpu_count()

class NoResultError(Exception):
    pass

class dbHnswIndexConfig(SQLModel, table=True):
    """
    Configuration associated with an hnswlib index as stored in the database
    """
    id:                 int       = Field(primary_key=True,                                               
                                          description='Unique database identifier for a given index')
    collection_id:      int       = Field(index=True, foreign_key="dbcollectionconfig.id", description="The Collection that the index is associated with.")
    count:              int       = Field(description="The number of vectors in the index.")
    filename:           str       = Field(description='The hnswlib index filename')
    md5sum:             str       = Field(description='The md5sum of the index file')
    M:                  int       = Field(ge=2, description="The M value passed to hnswlib when creating the index.")
    ef_construction:    int       = Field(ge=10, description="The ef_construction value passed to hnswlib when creating the index.")
    ef:                 int       = Field(ge=10, description="The recommended ef value to use at search time.")
    created_at:         float     = Field(default_factory=time, description='The epoch timestamp when the index was created.')
    hnsw_ix:   Optional[bytes]    = Field(sa_column=Column(LargeBinary), description='The actual hnswlib index.')
    

class dbCollectionConfig(SQLModel, table=True):
    """
    Configuration associated with a collection of strings, embeddings, as persisted in the database
    """
    id:              int             = Field(primary_key=True,                                               
                                             description='Unique database identifier for a given collection')    
    name:            str             = Field(index=True, description='The unique name of the collection')
    dim:             int             = Field(description='The dimensionality of the vectors in this collection')
    model:           Optional[str]   = Field(description='The (optional) name of the model used to embed the strings in this collection.')
    description:     Optional[str]   = Field(description='The (optional) description of this collection.')   
    created_at:      float           = Field(default_factory=time, description='The epoch timestamp when the collection was created.')
    
    @validator('name')
    def _name(cls, v):
        _is_valid_namestr(v, 'name')
        return v
    


class dbEmbedding(SQLModel, table=True):
    """
    An embedding as stored in the database
    """
    id:             int             = Field(primary_key=True,                                           
                                            description='Unique database identifier for a given embedding.')
    collection_id:  int             = Field(index=True, foreign_key="dbcollectionconfig.id", description="The Collection that the embedding is associated with.")
    doc_id:         Optional[str]   = Field(index=True, description="An optional user-specified document_id associated with the embedding.")
    vector:         bytes           = Field(sa_column=Column(LargeBinary), description='The user-supplied vector element as persisted in db as byte array. If sent in as a numpy array will be converted to bytes for storage.')
    text:           str             = Field(description="The text that was input to the model to generate this embedding.")
    meta:           Optional[str]   = Field(description="An optional json dictionary of metadata associated with the text.  Can be sent in as a dictionary and will be converted to json for storage.")
    created_at:     float           = Field(default_factory=time, description='The epoch timestamp when the embedding was created.')
    
    def __str__(self):
        if len(self.text) > 400:
            text = self.text[:400] + '...'
        else: 
            text = self.text
        text = text.replace('\n', ' ')
        estr = "Embedding("
        if self.doc_id is not None:
            estr += f"doc_id={self.doc_id}, "
        if self.meta is not None:
            estr += f"metadata={self.metadata_as_dict()}, "
        estr += f"text={text})"
        return estr
    
    def vector_as_array(self) -> np.array:    
        """
        return the stored vector as a numpy array        
        """
        return np.frombuffer(self.vector, dtype=np.float32)

    def vector_as_list(self) -> list[float]:    
        """
        return the stored vector as a numpy array        
        """
        return np.frombuffer(self.vector, dtype=np.float32).tolist()
    
    @root_validator(pre=True)
    def convert(cls, values):
        if 'vector' in values:
            if isinstance(values['vector'], np.ndarray):
                values['vector'] = values['vector'].astype(np.float32).tobytes()
        if 'meta' in values:
            if isinstance(values['meta'], dict):
                values['meta'] = dumps(values['meta'])
        return values
        
    def metadata_as_dict(self):        
        if self.meta is not None:
            return loads(self.meta)
            
    @classmethod
    def from_id(cls, 
                session: Session,
                id: int) -> "dbEmbedding":
        return session.get(dbEmbedding, id)



class Embedding(BaseModel):
    """
    An Embedding as sent to/from the Collection API
    """
    vector:         list[float]     = Field(description='The user-supplied vector element as stored as a list of floats. Can be sent in as a numpy array and will be converted to a list of floats.')   
    text:           str             = Field(description="The text that was input to the model to generate this embedding.")
    doc_id:         Optional[str]   = Field(description="An optional document_id associated with the embedding.")
    metadata:       Optional[dict]  = Field(description="An optional dictionary of metadata associated with the text")
    created_at:     float           = Field(default_factory=time, description='The epoch timestamp when the embedding was created.')

    def __str__(self) -> str:
        if len(self.text) > 400:
            text = self.text[:400] + '...'
        else: 
            text = self.text
        text = text.replace('\n', ' ')
        estr = "Embedding("
        if self.doc_id is not None:
            estr += f"doc_id={self.doc_id}, "
        if self.metadata is not None:
            estr += f"metadata={self.metadata}, "
        estr += f"text='{text}')"
        return estr
    
    @classmethod
    def from_db(cls, db_embedding: dbEmbedding) -> "Embedding":
        return Embedding(vector=db_embedding.vector_as_list(),
                         text=db_embedding.text,
                         doc_id=db_embedding.doc_id,
                         metadata=db_embedding.metadata_as_dict(),
                         created_at=db_embedding.created_at)
                         
    @root_validator(pre=True)
    def convert_vector(cls, values):
        if 'vector' in values:
            if isinstance(values['vector'], np.ndarray):
                # convert np.array to list 
                values['vector'] = values['vector'].tolist()
        return values

    def vector_as_array(self) -> np.array:    
        """
        return the stored vector as a numpy array        
        """
        return np.array(self.vector)

        
class SearchResponse(Embedding):
    """
    Return a search result consisting of an embedding and the distance to the query vector
    """
    distance: float


    
class Collection :
    """
    A combination of a sqlite database and an hnswlib index that provides a persistent collection of embeddings (strings+vectors+metadata) including hnswlib index configuration.
    """      
    def __init__(self, 
                 collection_name : str,
                 dimension       : int,                                  
                 sqlite_filename : Optional[str] = 'hnsqlite.sqlite',
                 modelname       : Optional[str] = None,
                 description     : Optional[str] = None) -> "Collection":
        """
        collection_name:    The name of the collection to use.  must be unique with the sqlite database            
        dimension:          The dimension of the vector space        
        sqlite_filename:    The name of the sqlite database file to use for the collection
                            If not specified a default name of 'hnsqlite.sqlite' will be used and the database will be created if it does not exist.
        modelname:          Optional name of the model used to generate the embeddings
        description:        Optional description of the collection
        
        Create a hnsqlite Collection object from a sqlite collection database file.
        Creates hnsqlite tables within the database if necessary.
        If the specified collection name is found in the database, the collection will be initialized
        from the database.  Otherwise a new collection of the specified name will be created in the database.
        """
        _is_valid_namestr(collection_name, 'name')
        if not os.path.exists(sqlite_filename):        
            logger.warning(f"sqlite_filename {sqlite_filename} does not exist; create new database")
        else:
            logger.info(f"sqlite_filename {sqlite_filename} exists; using existing database")
        db_engine = create_engine(f'sqlite:///{sqlite_filename}')
        SQLModel.metadata.create_all(db_engine)                    
        with Session(db_engine) as session:
            cconfig = session.exec(select(dbCollectionConfig).where(dbCollectionConfig.name == collection_name)).first()
            if cconfig:
                if cconfig.dim != dimension:
                    logger.error(f"collection {collection_name} already exists in {sqlite_filename} with dimension {cconfig.dim}")
                    raise ValueError(f"collection {collection_name} already exists in {sqlite_filename} with dimension {cconfig.dim}")
                else:
                    logger.info(f"using existing collection {collection_name} found in {sqlite_filename}")
            else:
                logger.warning(f"collection {collection_name} not found in {sqlite_filename}; creating new collection")
                cconfig = dbCollectionConfig(name=collection_name, 
                                             dim=dimension,
                                             model=modelname, 
                                             description=description)        
                with Session(db_engine) as session:            
                    session.add(cconfig)
                    session.commit()
                    session.refresh(cconfig)         
        
        self.config = cconfig
        self.db_engine = db_engine
        self.load_index()
   

    @classmethod
    def _save_index_to_disk(cls, name: str, hnsw_ix : dbHnswIndexConfig) -> Tuple[str, str, int]:
        """
        Save the current index to disk and return the filename, md5sum, and count of items in the index
        """
        count = len(hnsw_ix.get_ids_list())
        filename = f"index_{name}.hnsw"
        try:
            os.unlink(filename)
        except:
            pass
        hnsw_ix.save_index(filename)
        md5sum = md5_file(filename)
        logger.info(f"saved index to {filename} with md5sum {md5sum} and {count} items")
        return filename, md5sum, count

    def _save_index_to_db(self, index: dbHnswIndexConfig, delete_previous_index=True):
        """
        Save the current index config to the database
        """
        with Session(self.db_engine) as session:
            # delete old index from database
            if delete_previous_index:
                for old_index in session.query(dbHnswIndexConfig).filter(dbHnswIndexConfig.collection_id == self.config.id).all():
                    session.delete(old_index)
            session.add(index)
            session.commit()   # commit the new index record and deletion of the old index records
            session.refresh(index)
            logger.info(f"saved index to database with id {index.id}")

        
    def save_index(self, delete_previous_index=True) -> str:
        """
        Save the current hnsw index
        Used to persist the dbHnswIndexConfig after calling add_items or delete_items
        """
        # save the index to disk
        filename, md5sum, count = Collection._save_index_to_disk(self.config.name, self.hnsw_ix)

        # create the new index record
        index_config = dbHnswIndexConfig(collection_id = self.config.id,
                                        count = count,
                                        filename = filename,
                                        md5sum = md5sum,
                                        M = self.index_config.M,
                                        ef_construction = self.index_config.ef_construction,
                                        ef = self.index_config.ef_construction)
        
        self._save_index_to_db(index_config, delete_previous_index=delete_previous_index)
        return filename
    
    def make_index(self, M = 16, ef_construction = 200, delete_previous_index=True) -> dbHnswIndexConfig:
        """
        create an hnsw index that includes all embeddings in the collection database and use this new index for the collection going forward
        Returns the database representation of the index, not the hnswlib index object.
        This specific index can be loaded at a later time using the load_index method for testing purposes.
        """
        logger.info(f"make index for collection {self.config.name} with M={M}, ef_construction={ef_construction}")
        hnsw_ix = hnswlib.Index(space = 'cosine', dim = self.config.dim) 
        with Session(self.db_engine) as session:
            count = session.query(dbEmbedding).where(dbEmbedding.collection_id == self.config.id ).count()
        hnsw_ix.set_num_threads(1)   # filtering requires 1 thread only
        # TODO: predict good ef_construction, M, ef values based on count and dim
        hnsw_ix.init_index(max_elements = count, ef_construction = ef_construction, M = M, allow_replace_deleted=True)
        hnsw_ix.set_num_threads(CPU_COUNT)
        # add all elements to the index
        with Session(self.db_engine) as session:
            t0 = time()
            count = 0
            BATCH_SIZE = CPU_COUNT*16
            # batches process of embeddings for efficiency
            def process_batch(batch : list[dbEmbedding]):            
                # add batch of document embeddings to index
                hnsw_ix.add_items([e.vector for e in batch], [e.id for e in batch])            
                logger.info(f"hnsw_index.add_items   vectors/sec: {count/(time() - t0):.1f} ; total vectors: {count}")            
            batch = []            
            for de in session.exec(select(dbEmbedding).where(dbEmbedding.collection_id == self.config.id )).yield_per(BATCH_SIZE):
                batch.append(de)
                if len(batch) == BATCH_SIZE:
                    process_batch(batch)
                    count += len(batch)
                    batch = []
                    continue
            if batch:
                process_batch(batch)  # final partial batch
                count += len(batch)            
        hnsw_ix.set_num_threads(1)  # filtering requires 1 thread only  
        
        # persist the index to disk
        filename, md5sum, _count = Collection._save_index_to_disk(self.config.name, hnsw_ix)
        assert count == _count        
    
        # create the new index record
        index = dbHnswIndexConfig(collection_id = self.config.id,
                                count = count,
                                filename = filename,
                                md5sum = md5sum,
                                M = M,
                                ef_construction = ef_construction,
                                ef = ef_construction)

        self._save_index_to_db(index, delete_previous_index=delete_previous_index)         
        self.index_config = index     
        self.hnsw_ix = hnsw_ix                                  
        return index
    

    def load_index(self):
        """
        load the latest hnsw index from disk and use it for the collection.
        """
        logger.info(f"load index for {self.config}")        
        with Session(self.db_engine) as session:
            index_config = session.exec(select(dbHnswIndexConfig).where(dbHnswIndexConfig.collection_id == self.config.id).order_by(dbHnswIndexConfig.id.desc())).first()
            if not index_config:
                self.make_index()   # intialize and load the index
                return
        logger.info(f"loading index {index_config}")
        md5sum = md5_file(index_config.filename)
        if md5sum != index_config.md5sum:
            logger.error(f"md5sum {md5sum} does not match config.md5sum {index_config.md5sum}")
            self.make_index()   # intialize and load the index from scratch
            return
        else:
            logger.info(f"md5sum matches index.md5sum")
        hnsw_ix = hnswlib.Index(space='cosine', dim=self.config.dim)
        hnsw_ix.load_index(index_config.filename, max_elements=index_config.count, allow_replace_deleted=True)
        hnsw_ix.set_ef(index_config.ef)
        hnsw_ix.set_num_threads(1)   # filtering requires 1 thread only
        logger.info(f"load hnswlib index {index_config} complete")
        self.hnsw_ix = hnsw_ix
        self.index_config = index_config
        # validate that all expected vector IDs are in the index
        with Session(self.db_engine) as session:
            # get just the ids from the database
            embs = set(session.exec(select(dbEmbedding.id).where(dbEmbedding.collection_id == self.config.id )).all())
        missing_ids = embs - set(self.hnsw_ix.get_ids_list())
        if missing_ids:
            logger.info(f'updating index to include {len(missing_ids)} embedding ids found in db but not in index')
            embeddings = session.exec(select(dbEmbedding).where(dbEmbedding.id.in_(missing_ids))).all()
            count = self.hnsw_ix.get_current_count() + len(embeddings)
            self.hnsw_ix.resize_index(count)
            vectors = [e.vector_as_array() for e in embeddings]
            self.hnsw_ix.add_items(vectors, [e.id for e in embeddings], num_threads=CPU_COUNT, replace_deleted=True)      
            self.save_index()
            
          
    def __str__(self) -> str:
        cstr = f"Collection({self.config.name}, {self.config.dim}"
        if self.config.model:
            cstr += f", {self.config.model}"
        cstr += ")"
        return cstr

    def add_items(self,
                  vectors:        List[np.array],   # todo: support ndarray 
                  texts:          List[str],        
                  doc_ids:        Optional[List[str]] = None,
                  metadata:       Optional[List[dict]] = None,                 
                  save_index:     bool = True) -> None:             
        """
        add new items to the collection
        """                             
        if metadata is None:
            metadata = [None] * len(vectors)
        if doc_ids is None:
            doc_ids = [None] * len(vectors)

        # convert numpy vectors to bytes as float32 for storage in SQLite        
        vector_bytes = [v.astype(np.float32).tobytes() for v in vectors]
        cid = self.config.id
        embeddings = [dbEmbedding(vector=v, text=t, doc_id=d, meta=m, collection_id=cid) for v, t, d, m in zip(vector_bytes, texts, doc_ids, metadata)]      
        
        with Session(self.db_engine) as session:
            # Add new embeddings to the SQLite database
            session.add_all(embeddings)
            session.commit()

            # Refresh the embeddings to get their assigned IDs
            for e in embeddings:
                session.refresh(e)
                
        count = self.hnsw_ix.get_current_count() + len(vectors)
        self.hnsw_ix.resize_index(count)
        self.hnsw_ix.add_items(vectors, [e.id for e in embeddings], num_threads=CPU_COUNT, replace_deleted=True)
        if save_index:
            self.save_index()
            

    def add_embedding(self, embedding: Embedding, save_index=False) -> None:
        """
        add a single Embedding object to the collection
        """
        self.add_embeddings([embedding], save_index=save_index)
        
    def add_embeddings(self, embeddings: List[Embedding], save_index=False) -> None:
        """
        alternative to add_items that takes a list of Embedding objects
        """
        vectors = [e.vector_as_array() for e in embeddings]
        texts = [e.text for e in embeddings]
        doc_ids = [e.doc_id for e in embeddings]
        metadata = [e.metadata for e in embeddings]
        self.add_items(vectors, texts, doc_ids, metadata, save_index=save_index)

    def count(self) -> int:
        """
        return the number of items in the collection
        """
        with Session(self.db_engine) as session:
            return session.query(dbEmbedding).where(dbEmbedding.collection_id == self.config.id).count()

    def get_embeddings(self, start: int, limit: int, reverse :bool) -> List[Embedding]:
        """
        return a list of Embedding objects from the collection
        """
        with Session(self.db_engine) as session:    
            if reverse:
                query = select(dbEmbedding).where(dbEmbedding.collection_id == self.config.id).order_by(desc(dbEmbedding.id)).offset(start).limit(limit)
            else:
                query = select(dbEmbedding).where(dbEmbedding.collection_id == self.config.id).order_by(asc(dbEmbedding.id)).offset(start).limit(limit)                        
            return [Embedding.from_db(e) for e in session.exec(query)]

    def get_embeddings_doc_ids(self, doc_ids : List[str]) ->  List[Embedding]:
        """
        get the embeddings for the specified doc_ids
        """
        with Session(self.db_engine) as session:    
            query = select(dbEmbedding).where(dbEmbedding.collection_id == self.config.id).where(dbEmbedding.doc_id.in_(doc_ids))
            return [Embedding.from_db(e) for e in session.exec(query)]
        
    def search(self, vector: np.array, k = 12, filter=None) -> List[SearchResponse]:        
        """
        query the hnsw index for the nearest neighbors of the given vector
        unlike hnswlib which takes a 2D array of vectors, this method takes a single vector
        since the common use case for this in production is searching for a single vector at a time
        """
        if isinstance(vector, list):
            vector = np.array(vector)
        
        def _filter(id):
            with Session(self.db_engine) as session:
                # todo: replace with in-memory option
                e = session.exec(select(dbEmbedding).where(dbEmbedding.id == id)).first()                
                metadata = e.metadata_as_dict()
                result = filter_item(filter, metadata)                
                return result
        filter_func = _filter if filter else None

        
        # query the current index for the nearest neighbors
        while k > 0:            
            try:
                ids, distances = self.hnsw_ix.knn_query([vector], k=k, filter=filter_func)
                break
            except RuntimeError:
                # should verify there is at least 1 before commiting to this linear probe (or alternatively use a binary search)
                logger.info(f"hnsw_ix.knn_query failed with k={k}")
                k -= 1
                
        if k == 0:
            raise NoResultError("hnsw_ix.knn_query failed with k=0")

        # ids and distances are returned as 2D arrays, so we need to flatten them to 1D
        # and then zip them together to create a dict of id:distance
        ids = [int(i) for i in ids[0]]   # the embedding_ids    
        distances = [float(i) for i in distances[0]]  # the corresponding distances in increasing order        
        id_to_distance = dict(zip(ids, distances))

        # fetch the embeddings from the database by id
        # don't need to downselect/filter by collection id by since the embeding IDs are unique across collections
        with Session(self.db_engine) as session:
            embeddings = session.exec(select(dbEmbedding).where(dbEmbedding.id.in_(ids))).all()

        # create a list of SearchResponse objects sorted by distance
        responses = [SearchResponse(**Embedding.from_db(e).dict(), distance=id_to_distance[e.id]) for e in embeddings]
        responses.sort(key=lambda r: r.distance)
        return responses

    def delete(self, doc_ids : Optional[List[str]] = None, filter : Optional[dict] = None, delete_all : bool = False) -> None:
        """
        delete items from the collection based on doc_ids, items matching a filter, or everything
        """
        if delete_all:
            with Session(self.db_engine) as session:
                session.exec("delete from dbembedding")                
                session.commit()          
            self.make_index()  # create new index with no items
            logger.warning(f"deleted all items from {self.config.name}")
            return
        count = 0           
        if doc_ids:        
            logger.info(f"deleting doc_ids {doc_ids} from {self.config.name}")
            with Session(self.db_engine) as session:
                for embedding in session.exec(select(dbEmbedding).where(dbEmbedding.collection_id == self.config.id).where(dbEmbedding.doc_id.in_(doc_ids))):
                    count += 1
                    self.hnsw_ix.mark_deleted(embedding.id)
                    session.delete(embedding)
                session.commit()
        elif filter:
            logger.info(f"deleting items from {self.config.name} with filter: {filter}")
            with Session(self.db_engine) as session:
                for embedding in session.exec(select(dbEmbedding).where(dbEmbedding.collection_id == self.config.id)).all():
                    if filter_item(filter, embedding.metadata_as_dict()):
                        count += 1
                        self.hnsw_ix.mark_deleted(embedding.id)
                        session.delete(embedding)
                session.commit()
        if  count:
            logger.info(f"deleted {count} items from {self.config.name}")                   
            self.save_index()
