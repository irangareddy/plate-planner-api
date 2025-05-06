from neo4j import GraphDatabase

# Neo4j connection
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "database"
NEO4J_PASSWORD = "12345678"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Queries to inspect the database
queries = {
    "Node Labels": "CALL db.labels();",
    "Relationship Types": "CALL db.relationshipTypes();",
    "Node Counts by Label": "MATCH (n) RETURN labels(n)[0] AS label, COUNT(*) AS count ORDER BY count DESC;",
    "Relationship Counts by Type": "MATCH ()-[r]->() RETURN type(r) AS type, COUNT(*) AS count ORDER BY count DESC;",
    "Sample Ingredients": "MATCH (i:Ingredient) RETURN i.name LIMIT 20;",
    "Sample OCCURS_WITH Relationships": "MATCH (a:Ingredient)-[r:OCCURS_WITH]->(b:Ingredient) RETURN a.name AS from_ingredient, b.name AS to_ingredient, r LIMIT 20;",
    "Properties on OCCURS_WITH Relationships": "MATCH (a:Ingredient)-[r:OCCURS_WITH]->(b:Ingredient) RETURN keys(r) LIMIT 1;",
}

# Save the results into a file
with driver.session() as session, open("../data/support/db_snapshot.txt", "w", encoding="utf-8") as file:
    for title, query in queries.items():
        file.write(f"\n=== {title} ===\n")
        result = session.run(query)
        for record in result:
            file.write(str(record) + "\n")
        file.write("\n")

print("Database snapshot saved to db_snapshot.txt ✅")
