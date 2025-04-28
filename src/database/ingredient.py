from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "database"
NEO4J_PASSWORD = "12345678"

BATCH_SIZE = 200
MAX_WORKERS = 4  # Number of parallel threads; adjust based on your CPU and Neo4j capacity

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def batch_assign_weights():
    with driver.session() as session:
        total = session.execute_read(
            lambda tx: tx.run("MATCH (i:Ingredient) RETURN count(i) AS total").single()["total"])
        print(f"Total ingredients: {total}")

    batch_params = [(skip, BATCH_SIZE) for skip in range(0, total, BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_batch, skip, limit) for skip, limit in batch_params]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Batch failed: {e}")


def process_batch(skip, limit):
    with driver.session() as session:
        session.execute_write(run_batch_query, skip, limit)
        print(f"Completed batch: skip {skip} limit {limit}")


def run_batch_query(tx, skip, limit):
    tx.run("""
        MATCH (a:Ingredient)
        WITH a
        ORDER BY id(a)
        SKIP $skip LIMIT $limit
        MATCH (a)-[:OCCURS_WITH]-(common)-[:OCCURS_WITH]-(b:Ingredient)
        WHERE id(a) < id(b)
        WITH a, b, count(common) AS commonNeighbors
        MERGE (a)-[r:OCCURS_WITH]->(b)
        SET r.weight = commonNeighbors
    """, skip=skip, limit=limit)


if __name__ == "__main__":
    batch_assign_weights()
    driver.close()
