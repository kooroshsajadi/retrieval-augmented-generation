from rdflib import Graph, URIRef, Namespace, Literal, XSD
import csv

def get_csv_columns(csv_path):
    """
    Reads the header row of a CSV file and returns the column names.

    Args:
        csv_path (str): Path to the input CSV file.

    Returns:
        List[str]: List of column names.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
    return headers

def csv_to_rdf(csv_path: str, rdf_output_path: str):
    """
    Reads a GeoNames CSV file, converts each row to RDF triples, and saves as Turtle.
    Args:
        csv_path (str): Path to the input CSV file.
        rdf_output_path (str): Path where the RDF Turtle file will be saved.
    """
    ns = Namespace("http://www.geonames.org/ontology#")
    g = Graph()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = URIRef(ns + row["geonameid"])
            g.add((subject, ns.name, Literal(row["name"])))
            g.add((subject, ns.asciiname, Literal(row["asciiname"])))
            g.add((subject, ns.alternatenames, Literal(row["alternatenames"])))
            g.add((subject, ns.latitude, Literal(row["latitude"], datatype=XSD.float)))
            g.add((subject, ns.longitude, Literal(row["longitude"], datatype=XSD.float)))
            g.add((subject, ns.featureClass, Literal(row["feature class"])))
            g.add((subject, ns.featureCode, Literal(row["feature code"])))
            g.add((subject, ns.countryCode, Literal(row["country code"])))
            g.add((subject, ns.cc2, Literal(row["cc2"])))
            g.add((subject, ns.admin1Code, Literal(row["admin1 code"])))
            g.add((subject, ns.admin2Code, Literal(row["admin2 code"])))
            g.add((subject, ns.admin3Code, Literal(row["admin3 code"])))
            g.add((subject, ns.admin4Code, Literal(row["admin4 code"])))
            g.add((subject, ns.population, Literal(row["population"], datatype=XSD.integer)))
            g.add((subject, ns.elevation, Literal(row["elevation"], datatype=XSD.integer)))
            g.add((subject, ns.dem, Literal(row["dem"], datatype=XSD.integer)))
            g.add((subject, ns.timezone, Literal(row["timezone"])))
            g.add((subject, ns.modificationDate, Literal(row["modification date"], datatype=XSD.date)))
    g.serialize(destination=rdf_output_path, format="turtle")
    print(f"RDF data saved to {rdf_output_path}")


def main():
    csv_to_rdf("geonames.csv", "geonames.ttl")
    # columns = get_csv_columns("geonames.csv")
    # print(columns)


if __name__ == "__main__":
    main()


