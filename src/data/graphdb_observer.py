import requests

class GraphDBObserver:
    def __init__(self, graphdb_url, username=None, password=None):
        """
        Initialize the observer.
        :param graphdb_url: Base URL of the GraphDB instance (e.g., "http://localhost:7200").
        :param username: If authentication is required.
        :param password: If authentication is required.
        """
        self.graphdb_url = graphdb_url.rstrip("/")
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)

    def list_repositories(self):
        """
        Fetch the list of repositories from GraphDB.
        """
        url = f"{self.graphdb_url}/rest/repositories"
        response = self.session.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        data = response.json()
        return [repo['id'] for repo in data]

    def list_graphs(self, repository_id):
        """
        List named graphs (contexts) in the repository.
        :param repository_id: Repository ID as shown in list_repositories().
        """
        url = f"{self.graphdb_url}/repositories/{repository_id}/contexts"
        response = self.session.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        contexts = response.json().get('results', {}).get('bindings', [])
        return [c['contextID']['value'] for c in contexts]
    
    def print_graph_structure(self, repository_id: str, graph_uri: str, sample_triples: int = 5):
        """
        Extracts and prints essential structure information about a named graph in a GraphDB repository.

        :param repository_id: The ID of the GraphDB repository.
        :param graph_uri: The full URI of the named graph (context).
        :param sample_triples: How many sample triples to print for review.
        """
        endpoint = f"{self.graphdb_url}/repositories/{repository_id}"

        def sparql(query):
            resp = self.session.post(
                endpoint,
                data={"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()["results"]["bindings"]

        # 1. Total triples in the graph
        triples = sparql(f"""
            SELECT (COUNT(*) as ?count)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
        """)[0]['count']['value']

        # 2. Unique counts for S, P, O
        unique_subjects = sparql(f"""
            SELECT (COUNT(DISTINCT ?s) as ?count)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
        """)[0]['count']['value']

        unique_predicates = sparql(f"""
            SELECT (COUNT(DISTINCT ?p) as ?count)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
        """)[0]['count']['value']

        unique_objects = sparql(f"""
            SELECT (COUNT(DISTINCT ?o) as ?count)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
        """)[0]['count']['value']

        # 3. Classes used (rdf:type)
        classes = sparql(f"""
            SELECT DISTINCT ?class
            WHERE {{ GRAPH <{graph_uri}> {{ ?s a ?class }} }}
        """)
        class_list = [c['class']['value'] for c in classes]

        # 4. Namespaces (of predicates)
        namespaces = sparql(f"""
            SELECT DISTINCT (STRBEFORE(STR(?p), "#") as ?ns)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} FILTER(CONTAINS(STR(?p), "#")) }}
        """)
        ns_list = list({n['ns']['value'] for n in namespaces if n.get('ns', {}).get('value')})

        # 5. Datatypes of literals
        datatypes = sparql(f"""
            SELECT DISTINCT (DATATYPE(?o) AS ?datatype)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o . FILTER(isLiteral(?o)) }} }}
        """)
        datatype_list = [d['datatype']['value'] for d in datatypes if d.get('datatype', {}).get('value')]

        # 6. Most frequent predicates
        predicates = sparql(f"""
            SELECT ?p (COUNT(*) as ?count)
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
            GROUP BY ?p ORDER BY DESC(?count) LIMIT 5
        """)
        frequent_predicates = [(p['p']['value'], int(p['count']['value'])) for p in predicates]

        # 7. Sample triples
        samples = sparql(f"""
            SELECT ?s ?p ?o
            WHERE {{ GRAPH <{graph_uri}> {{ ?s ?p ?o }} }}
            LIMIT {sample_triples}
        """)

        # 8. Graph metadata (optional)
        metadata = sparql(f"""
            SELECT ?label ?comment
            WHERE {{
                GRAPH <{graph_uri}> {{
                    OPTIONAL {{ <{graph_uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?label }}
                    OPTIONAL {{ <{graph_uri}> <http://www.w3.org/2000/01/rdf-schema#comment> ?comment }}
                }}
            }}
            LIMIT 1
        """)
        meta = metadata[0] if metadata else {}

        # --- Print Section ---
        print(f"\nGraphDB Structure for Graph: {graph_uri}")
        print(f"Total triples: {triples}")
        print(f"Unique subjects: {unique_subjects}")
        print(f"Unique predicates: {unique_predicates}")
        print(f"Unique objects: {unique_objects}")
        print(f"Classes used: {class_list}")
        print(f"Namespaces (predicates): {ns_list}")
        print(f"Data types used: {datatype_list}")
        print("Most frequent predicates:")
        for uri, cnt in frequent_predicates:
            print(f"  {uri} ({cnt} occurrences)")
        print(f"\nSample triples (max {sample_triples}):")
        for s in samples:
            s_val = s['s']['value']
            p_val = s['p']['value']
            o_val = s['o']['value']
            print(f"  {s_val} {p_val} {o_val}")
        if meta.get('label', {}).get('value') or meta.get('comment', {}).get('value'):
            print("Metadata:")
            if meta.get('label', {}).get('value'):
                print("  rdfs:label:", meta['label']['value'])
            if meta.get('comment', {}).get('value'):
                print("  rdfs:comment:", meta['comment']['value'])
        print()

def main():
    # Example usage (edit these values as needed)
    GRAPHDB_URL = "http://localhost:7200"  # Change to your GraphDB URL
    USERNAME = None       # Optional
    PASSWORD = None       # Optional

    observer = GraphDBObserver(GRAPHDB_URL, USERNAME, PASSWORD)
    
    # Decide what to do
    repositories = observer.list_repositories()
    print("Repositories:", repositories)

    # For demonstration: List graphs in the first repository
    if repositories:
        graphs = observer.list_graphs(repositories[0])
        print(f"Graphs in repository '{repositories[0]}':", graphs)

    observer.print_graph_structure('rag', 'http://example.org/foaf', 10)

if __name__ == "__main__":
    main()
