from pymilvus import Collection, connections, utility

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# List collections.
print("Collections:", utility.list_collections())

# Drop the collection
# utility.drop_collection("test_gotmat_collection")
# print("Dropped collection: test_gotmat_collection")

# Get the collection
collection = Collection("gotmat_collection")

# Get the number of entities in the collection
num_entities = collection.num_entities
print(f"Number of entities in the collection: {num_entities}")
