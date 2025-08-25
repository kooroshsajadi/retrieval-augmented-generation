from pymilvus import MilvusClient

client = MilvusClient("gotmat_client.db")

if client.has_collection(collection_name="gotmat_collection"):
    client.drop_collection(collection_name="gotmat_collection")
client.create_collection(
    collection_name="gotmat_collection",
    dimension=768,  # TODO: Check your case
)
