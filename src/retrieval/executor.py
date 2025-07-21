from src.retrieval.connection import get_graphdb_connection

def execute_query(endpoint_url, query):
    """
    Executes a SPARQL query at the provided endpoint.

    Args:
        endpoint_url (str): GraphDB repository SPARQL endpoint.
        query (str): The SPARQL query string.

    Returns:
        list of dict: Each dict represents a row, with variable bindings as keys.
    """
    conn = get_graphdb_connection(endpoint_url)
    conn.setQuery(query)
    results = conn.query().convert()['results']['bindings']
    # Parse results to clean dicts for direct use
    parsed = []
    for row in results:
        entry = {k: v['value'] for k, v in row.items()}
        parsed.append(entry)
    return parsed

if __name__ == "__main__":
    # Example parameters
    endpoint_url = "http://localhost:7200/repositories/rag"

    # Load SPARQL query from a file
    query_file = "src/data/queries/list_entity_types.rq"
    with open(query_file, "r", encoding="utf-8") as f:
        query = f.read()

    results = execute_query(endpoint_url, query)

    # Print results in a readable way
    if results:
        # Print as columns
        keys = results[0].keys()
        print("\t".join(keys))
        for row in results:
            print("\t".join([row.get(k, "") for k in keys]))
    else:
        print("No results found.")
