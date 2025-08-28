from pymilvus import connections, utility, Collection

connections.connect(alias="default", host="localhost", port="19530")
print("Collections:", utility.list_collections())

collection = Collection("gotmat_collection")
print("Number of entities in collection:", collection.num_entities)