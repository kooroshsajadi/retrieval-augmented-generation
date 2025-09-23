from pymilvus import connections, utility

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# List collections before deletion
print("Collections before removal:", utility.list_collections())

# Drop the collection
# utility.drop_collection("test_gotmat_collection")
# print("Dropped collection: test_gotmat_collection")

# List collections after deletion to confirm removal
print("Collections after removal:", utility.list_collections())
