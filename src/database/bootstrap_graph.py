from src.database import add_edges_from_csv, build_similar_to_edges, explore_util, load_into_neo4j


def bootstrap():
    print("\n🚀 Step 1: Loading ingredients, recipes, and relationships...")
    load_into_neo4j.main()

    print("\n🔗 Step 2: Adding SUBSTITUTES_WITH edges...")
    add_edges_from_csv.main()

    print("\n🔁 Step 3: Building SIMILAR_TO relationships...")
    build_similar_to_edges.main()

    print("\n🔍 Step 4: Running Neo4j exploration summary...")
    explore_util.main()

    print("\n✅ Graph bootstrap complete.")


if __name__ == "__main__":
    bootstrap()
