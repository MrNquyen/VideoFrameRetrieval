from pymilvus import MilvusClient,  DataType
from typing import List
# https://milvus.io/docs/install_standalone-docker-compose.md
class Milvus:
    def __init__(self, milvus_config):
        self.client = MilvusClient(milvus_config["uri"], milvus_config["token"])
        self.database = self.create_database(milvus_config["database_name"])
        self.config = milvus_config

    def create_database(self, database_name):
        if database_name not in self.client.list_databases():
            self.client.create_database(database_name)
            print(f"Database '{database_name}' created.")
        else:
            print(f"Database '{database_name}' already exists.")
    

    def create_collection(self):
        """
            1. Create a schema for the Milvus database with the following fields:
            2. Create indexing
            3. Create a collection with the schema
        """
        schema_config = self.config['schema_config']
        # 1
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=schema_config["id_max_length"], is_primary=True)
        schema.add_field(field_name=schema_config["visual_col_name"], datatype=DataType.FLOAT_VECTOR, dim=schema_config['visual_features_dim'])
        schema.add_field(field_name=schema_config["embedding_col_name"], datatype=DataType.FLOAT_VECTOR, dim=schema_config['text_features_dim'])
        schema.add_field(field_name=schema_config["objects_col_name"], datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=schema_config['max_objects'], nullable=True, max_length=schema_config['object_element_max_length'])
        schema.add_field(field_name=schema_config["concepts_col_name"], datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=schema_config['max_objects'], nullable=True, max_length=schema_config['concept_element_max_length'])
        schema.add_field(field_name=schema_config["timestamp_col_name"], datatype=DataType.VARCHAR, max_length=schema_config["timestamp_max_length"], nullable=True)

        # 2.1 vector indexing
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name = self.config["visual_col_name"],
            metric_type = 'COSINE',
            index_type = 'IVF_FLAT',
            index_name= 'visual_features_index',
            params = {
                "nlist": schema_config['nlist']
            }
        )

        index_params.add_index(
            field_name = self.config["embedding_col_name"],
            metric_type = 'COSINE',
            index_type = 'IVF_FLAT',
            index_name= 'mm_caption_embeddings_index',
            params = {
                "nlist": schema_config['nlist']
            }
        )

        # 2.2 scalar index

        index_params.add_index(
            field_name = self.config["objects_col_name"],
            index_type = 'INVERTED',
            index_name= 'objects_index',
        )

        index_params.add_index(
            field_name = self.config["concepts_col_name"],
            index_type = 'INVERTED',
            index_name= 'concepts_index',
        )

        # 3

        collection_name = schema_config['collection_name']
        if not self.client.has_collection(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"Collection '{collection_name}' created with schema and indexes.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    
    def combine_filters(self, *filters):
        """
            Kết hợp các filter thành một chuỗi AND
        """
        combined_filter = " AND ".join(filters)
        return combined_filter
    
    def filter_scalar(self, embedding_queries, objects_queries: list, concepts_queries: list, topk : int):
        object_filter = f"ARRAY_CONTAINS_ALL({self.config['objects_col_name']}, {objects_queries})"
        concept_filter = f"ARRAY_CONTAINS_ALL({self.config['concepts_col_name']}, {concepts_queries})"
        
        combined_filter = self.combine_filters(object_filter, concept_filter)

        return self.client.search(
            collection_name=self.config['collection_name'],
            anns_field=self.config["embedding_col_name"],
            data=embedding_queries,
            filter=combined_filter,
            output_fields=[self.config["visual_col_name"], self.config["embedding_col_name"], self.config["objects_col_name"], self.config["concepts_col_name"], self.config["timestamp_col_name"]],
            limit=topk
        )

    def find_similar_vector(self, embedding_queries, top_k=10):
        """
            Cho 1 / N vector query, tìm kiếm các vector mm embed tương tự trong collection
        """
        return self.client.search(
            collection_name=self.config['collection_name'],
            anns_field=self.config["embedding_col_name"],
            data=embedding_queries,
            output_fields=[self.config["visual_col_name"], self.config["embedding_col_name"], self.config["objects_col_name"], self.config["concepts_col_name"], self.config["timestamp_col_name"]],
            limit=top_k
        )
    
    
    def insert(self, data):
        """
            Chèn dữ liệu vào collection
        """
        return self.client.insert(
            collection_name=self.config['collection_name'],
            data=data
        )


class MilvusLight:
    def __init__(self, milvus_config):
        self.client = MilvusClient("milvus_demo.db")
        self.database = self.create_database("milvus_demo.db")
        self.config = milvus_config

    def create_database(self, database_name):
        if database_name not in self.client.list_databases():
            self.client.create_database(database_name)
            print(f"Database '{database_name}' created.")
        else:
            print(f"Database '{database_name}' already exists.")
    

    def create_collection(self):
        """
            1. Create a schema for the Milvus database with the following fields:
            2. Create indexing
            3. Create a collection with the schema
        """
        schema_config = self.config['schema_config']
        # 1
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=schema_config["id_max_length"], is_primary=True)
        schema.add_field(field_name=schema_config["visual_col_name"], datatype=DataType.FLOAT_VECTOR, dim=schema_config['visual_features_dim'])
        schema.add_field(field_name=schema_config["embedding_col_name"], datatype=DataType.FLOAT_VECTOR, dim=schema_config['text_features_dim'])
        schema.add_field(field_name=schema_config["objects_col_name"], datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=schema_config['max_objects'], nullable=True, max_length=schema_config['object_element_max_length'])
        schema.add_field(field_name=schema_config["concepts_col_name"], datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=schema_config['max_objects'], nullable=True, max_length=schema_config['concept_element_max_length'])
        schema.add_field(field_name=schema_config["timestamp_col_name"], datatype=DataType.VARCHAR, max_length=schema_config["timestamp_max_length"], nullable=True)

        # 2.1 vector indexing
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name = self.config["visual_col_name"],
            metric_type = 'COSINE',
            index_type = 'IVF_FLAT',
            index_name= 'visual_features_index',
            params = {
                "nlist": schema_config['nlist']
            }
        )

        index_params.add_index(
            field_name = self.config["embedding_col_name"],
            metric_type = 'COSINE',
            index_type = 'IVF_FLAT',
            index_name= 'mm_caption_embeddings_index',
            params = {
                "nlist": schema_config['nlist']
            }
        )

        # 2.2 scalar index

        index_params.add_index(
            field_name = self.config["objects_col_name"],
            index_type = 'INVERTED',
            index_name= 'objects_index',
        )

        index_params.add_index(
            field_name = self.config["concepts_col_name"],
            index_type = 'INVERTED',
            index_name= 'concepts_index',
        )

        # 3

        collection_name = schema_config['collection_name']
        if not self.client.has_collection(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"Collection '{collection_name}' created with schema and indexes.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    
    def combine_filters(self, *filters):
        """
            Kết hợp các filter thành một chuỗi AND
        """
        combined_filter = " AND ".join(filters)
        return combined_filter
    
    def filter_scalar(self, embedding_queries, objects_queries: list, concepts_queries: list, topk : int):
        object_filter = f"ARRAY_CONTAINS_ALL({self.config['objects_col_name']}, {objects_queries})"
        concept_filter = f"ARRAY_CONTAINS_ALL({self.config['concepts_col_name']}, {concepts_queries})"
        
        combined_filter = self.combine_filters(object_filter, concept_filter)

        return self.client.search(
            collection_name=self.config['collection_name'],
            anns_field=self.config["embedding_col_name"],
            data=embedding_queries,
            filter=combined_filter,
            output_fields=[self.config["visual_col_name"], self.config["embedding_col_name"], self.config["objects_col_name"], self.config["concepts_col_name"], self.config["timestamp_col_name"]],
            limit=topk
        )

    def find_similar_vector(self, embedding_queries, top_k=10):
        """
            Cho 1 / N vector query, tìm kiếm các vector mm embed tương tự trong collection
        """
        return self.client.search(
            collection_name=self.config['collection_name'],
            anns_field=self.config["embedding_col_name"],
            data=embedding_queries,
            output_fields=[self.config["visual_col_name"], self.config["embedding_col_name"], self.config["objects_col_name"], self.config["concepts_col_name"], self.config["timestamp_col_name"]],
            limit=top_k
        )
    
    
    def insert(self, data):
        """
            Chèn dữ liệu vào collection
        """
        return self.client.insert(
            collection_name=self.config['collection_name'],
            data=data
        )
    

class MilvusLightV2:
    def __init__(self, milvus_config, uri: str = "http://127.0.0.1:19530"):
        """
        milvus_config: dict containing at least a 'schema_config' dict and other config keys used below.
        uri: Milvus server uri (e.g. "http://127.0.0.1:19530" or cloud uri with token).
        """
        self.config = milvus_config
        # create client with explicit uri
        self.client = MilvusClient(uri=uri)
        # ensure database exists (best-effort)
        self.database = self.create_database(self.config.get("database_name", "milvus_demo.db"))

    def create_database(self, database_name: str):
        """
        Try to ensure the database exists. Some Milvus server versions may not implement
        list_databases(); handle that gracefully by attempting create_database().
        Returns True if database exists (or was created), False otherwise.
        """
        try:
            # not all servers implement list_databases; catch failures
            dbs = self.client.list_databases()
            if database_name not in dbs:
                self.client.create_database(database_name)
                print("Database '%s' created.", database_name)
            else:
                print("Database '%s' already exists.", database_name)
            return True
        except Exception as e:
            # If list_databases is unimplemented, attempt to create and continue.
            print("list_databases() failed (%s). Attempting create_database() directly.", e)
            try:
                self.client.create_database(database_name)
                print("Database '%s' created via fallback.", database_name)
                return True
            except Exception as e2:
                print("Failed to create or confirm database '%s': %s", database_name, e2)
                return False

    def create_collection(self):
        """
        1) Build schema from config
        2) Prepare index params
        3) Create collection if not exists with schema & index_params
        """
        schema_config = self.config["schema_config"]

        # 1. Schema
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            max_length=schema_config["id_max_length"],
            is_primary=True,
        )
        schema.add_field(
            field_name=schema_config["visual_col_name"],
            datatype=DataType.FLOAT_VECTOR,
            dim=schema_config["visual_features_dim"],
        )
        schema.add_field(
            field_name=schema_config["embedding_col_name"],
            datatype=DataType.FLOAT_VECTOR,
            dim=schema_config["text_features_dim"],
        )
        schema.add_field(
            field_name=schema_config["objects_col_name"],
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=schema_config["max_objects"],
            nullable=True,
            max_length=schema_config["object_element_max_length"],
        )
        schema.add_field(
            field_name=schema_config["concepts_col_name"],
            datatype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=schema_config["max_objects"],
            nullable=True,
            max_length=schema_config["concept_element_max_length"],
        )
        schema.add_field(
            field_name=schema_config["timestamp_col_name"],
            datatype=DataType.VARCHAR,
            max_length=schema_config["timestamp_max_length"],
            nullable=True,
        )

        # 2. Index params (vector + scalar)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=schema_config["visual_col_name"],
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="visual_features_index",
            params={"nlist": schema_config["nlist"]},
        )
        index_params.add_index(
            field_name=schema_config["embedding_col_name"],
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="mm_caption_embeddings_index",
            params={"nlist": schema_config["nlist"]},
        )
        # scalar inverted indexes for array fields
        index_params.add_index(
            field_name=schema_config["objects_col_name"],
            index_type="INVERTED",
            index_name="objects_index",
        )
        index_params.add_index(
            field_name=schema_config["concepts_col_name"],
            index_type="INVERTED",
            index_name="concepts_index",
        )

        # 3. Create collection if not exists
        collection_name = schema_config["collection_name"]
        try:
            if not self.client.has_collection(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    schema=schema,
                    index_params=index_params,
                )
                print("Collection '%s' created with schema and indexes.", collection_name)
            else:
                print("Collection '%s' already exists.", collection_name)
            return True
        except Exception as e:
            print("Failed to create collection '%s': %s", collection_name, e)
            return False

    def combine_filters(self, *filters: str) -> str:
        """Combine multiple filter clauses with AND, ignoring empty ones."""
        filters = [f for f in filters if f and f.strip()]
        return " AND ".join(filters) if filters else ""

    def filter_scalar(self, embedding_queries, objects_queries: list = None, concepts_queries: list = None, topk: int = 10):
        """
        Search by embedding and optionally filter by objects/concepts arrays.
        objects_queries and concepts_queries should be lists of strings (e.g. ['dog','car']).
        """
        schema_cfg = self.config["schema_config"]
        filters = []

        if objects_queries:
            obj_json = json.dumps(objects_queries, ensure_ascii=False)
            filters.append(f'ARRAY_CONTAINS_ALL({schema_cfg["objects_col_name"]}, {obj_json})')

        if concepts_queries:
            con_json = json.dumps(concepts_queries, ensure_ascii=False)
            filters.append(f'ARRAY_CONTAINS_ALL({schema_cfg["concepts_col_name"]}, {con_json})')

        combined_filter = self.combine_filters(*filters)

        try:
            return self.client.search(
                collection_name=schema_cfg["collection_name"],
                anns_field=schema_cfg["embedding_col_name"],
                data=embedding_queries,
                filter=combined_filter or None,
                output_fields=[
                    schema_cfg["visual_col_name"],
                    schema_cfg["embedding_col_name"],
                    schema_cfg["objects_col_name"],
                    schema_cfg["concepts_col_name"],
                    schema_cfg["timestamp_col_name"],
                ],
                limit=topk,
            )
        except Exception:
            print("filter_scalar search failed. filter=%s", combined_filter)
            raise

    def find_similar_vector(self, embedding_queries, top_k: int = 10):
        """Search nearest neighbors by the embedding vector field."""
        schema_cfg = self.config["schema_config"]
        try:
            return self.client.search(
                collection_name=schema_cfg["collection_name"],
                anns_field=schema_cfg["embedding_col_name"],
                data=embedding_queries,
                output_fields=[
                    schema_cfg["visual_col_name"],
                    schema_cfg["embedding_col_name"],
                    schema_cfg["objects_col_name"],
                    schema_cfg["concepts_col_name"],
                    schema_cfg["timestamp_col_name"],
                ],
                limit=top_k,
            )
        except Exception:
            print("find_similar_vector failed.")
            raise

    def insert(self, data, ids=None):
        """
        Insert data into collection.
        `data` must match the collection schema (list-of-columns, or list-of-dicts depending on your usage).
        Optionally pass ids.
        """
        collection_name = self.config["schema_config"]["collection_name"]
        try:
            return self.client.insert(collection_name=collection_name, data=data, ids=ids)
        except Exception:
            print("Insert failed.")
            raise

import random

milvus_config = {
    "uri": "http://localhost:19530",
    "token": "",
    "database_name": "test_db",
    "collection_name": "test_collection",
    
    # Các tên field
    "visual_col_name": "visual_features",
    "embedding_col_name": "mm_caption_embeddings",
    "objects_col_name": "objects",
    "concepts_col_name": "concepts",
    "timestamp_col_name": "timestamp",

    # ⚠️ Các config schema nằm trong `schema_config`
    "schema_config": {
        "collection_name": "test_collection",
        "id_max_length": 64,
        "visual_features_dim": 128,
        "text_features_dim": 128,
        "max_objects": 5,
        "timestamp_max_length": 32,
        "nlist": 128,
        "visual_col_name": "visual_features",
        "embedding_col_name": "mm_caption_embeddings",
        "objects_col_name": "objects",
        "concepts_col_name": "concepts",
        "timestamp_col_name": "timestamp",
        "object_element_max_length": 32,
        "concept_element_max_length": 32
    }
}

def insert_dummy_data(
        milvus_instance,
        features_dict: List[dict]
    ):
    """
        Chèn dữ liệu giả vào collection để test
    """
    data = []
    num_samples = len(features_dict)
    for item_features in features_dict:
        item = {
            "visual_feature": None,
            "text_feature": None,
            "objects": [],
            "concepts": [],
            "timestamp": None,
        }

        data.append({
            "id": item_features["id"],
            milvus_instance.config["visual_col_name"]: item_features["visual_feature"],
            milvus_instance.config["embedding_col_name"]: item_features["text_feature"],
            milvus_instance.config["objects_col_name"]: item_features["objects"],
            milvus_instance.config["concepts_col_name"]: item_features["concepts"],
            milvus_instance.config["timestamp_col_name"]: item_features["timestamp"]
        })
    
    milvus_instance.insert(data)
    print(f"Inserted {num_samples} dummy data samples.")


# --- TẠO CLASS VÀ COLLECTION ---
# milvus_instance = Milvus(milvus_config)
# milvus_instance.create_collection()

# # --- CHÈN DỮ LIỆU GIẢ ---
# insert_dummy_data(milvus_instance, num_samples=30)
# import numpy as np
# # --- TEST 1: VECTOR SEARCH ---
# query_vector = [np.random.rand(128).tolist()]  # 1 vector query
# similar_results = milvus_instance.find_similar_vector(query_vector)
# print("\n🎯 Similar Vector Search Results:")
# for item in similar_results:
#     print(item)

# # --- TEST 2: FILTER SCALAR + VECTOR ---
# objects_query = ["dog", "car"]
# concepts_query = ["urban"]
# filtered_results = milvus_instance.filter_scalar(query_vector, objects_query, concepts_query, topk=5)
# print("\n🧪 Filtered Scalar + Vector Search Results:")
# for item in filtered_results:
#     print(item)