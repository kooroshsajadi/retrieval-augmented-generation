from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

#TODO: Refactor class to be similar to Milvus connector.
# TODO: Add an interface for connection.

class GraphDBConnectionError(Exception):
    """Custom exception for connection failures."""
    pass

def get_graphdb_connection(endpoint_url, test_query=None, timeout=10):
    """
    Establishes and validates a connection to the specified GraphDB SPARQL endpoint.

    Args:
        endpoint_url (str): Full URL of the GraphDB SPARQL endpoint.
        test_query (str, optional): Custom SPARQL query for validation. Defaults to None (uses ASK query).
        timeout (int, optional): Timeout in seconds for query execution.

    Returns:
        SPARQLWrapper: Configured and validated SPARQL connection object.

    Raises:
        GraphDBConnectionError: If the connection or query fails.
    """
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)

    # Use a harmless query to validate connection
    validation_query = test_query or "ASK {}"
    try:
        sparql.setQuery(validation_query)
        response = sparql.query().convert()
        # Successful response: for ASK {}, expect JSON: {'boolean': True/False}
        if (isinstance(response, dict) and
            ('boolean' in response or 'head' in response)):
            return sparql
        else:
            raise GraphDBConnectionError(
                f"Unexpected response format from GraphDB endpoint: {response}"
            )
    except (Exception, SPARQLExceptions.EndPointNotFound) as err:
        raise GraphDBConnectionError(
            f"Failed to connect or validate endpoint '{endpoint_url}': {err}"
        )

# Example usage (uncomment for testing):
# if __name__ == "__main__":
#     endpoint = "http://localhost:7200/repositories/rag"
#     try:
#         conn = get_graphdb_connection(endpoint)
#         print("Connection validated successfully!")
#     except GraphDBConnectionError as e:
#         print(str(e))
