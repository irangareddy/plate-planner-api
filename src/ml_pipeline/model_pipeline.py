import pandas as pd
from gensim.models import Word2Vec
from neo4j import GraphDatabase
from typing import List, Tuple

def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['ingredients_list'] = df['NER'].apply(eval)  # assuming NER is a stringified list
    return df

def train_word2vec(ingredient_sentences: List[List[str]], vector_size: int = 128, window: int = 5, min_count: int = 5) -> Word2Vec:
    model = Word2Vec(
        sentences=ingredient_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1  # skip-gram model
    )
    return model

def find_similar_ingredients(model: Word2Vec, topn: int = 5, similarity_threshold: float = 0.6) -> List[Tuple[str, str, float]]:
    substitution_pairs = []
    for ingredient in model.wv.index_to_key:
        try:
            similars = model.wv.most_similar(ingredient, topn=topn)
            for similar_ing, score in similars:
                if score >= similarity_threshold:
                    substitution_pairs.append((ingredient, similar_ing, score))
        except KeyError:
            continue
    return substitution_pairs

def generate_cypher(substitution_pairs: List[Tuple[str, str, float]]) -> List[str]:
    cypher_statements = []
    for ing1, ing2, score in substitution_pairs:
        cypher = f'''
        MATCH (a:Ingredient {{name: "{ing1}"}}), (b:Ingredient {{name: "{ing2}"}})
        MERGE (a)-[r:SUBSTITUTES_WITH]->(b)
        SET r.context = "general", r.similarityScore = {score:.4f};
        '''
        cypher_statements.append(cypher)
    return cypher_statements

def push_to_neo4j(cypher_statements: List[str], uri: str, user: str, password: str):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def execute_query(tx, query):
        tx.run(query)

    with driver.session() as session:
        for query in cypher_statements:
            session.write_transaction(execute_query, query)

    driver.close()

def run_pipeline(
    recipe_filepath: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str
):
    # Step 1: Load and Parse
    df = load_dataset(recipe_filepath)
    print(f"Loaded {len(df)} recipes.")

    # Step 2: Train Word2Vec
    ingredient_sentences = df['ingredients_list'].tolist()
    w2v_model = train_word2vec(ingredient_sentences)
    print(f"Trained Word2Vec model on {len(w2v_model.wv.index_to_key)} unique ingredients.")

    # Step 3: Find Similar Ingredients
    substitution_pairs = find_similar_ingredients(w2v_model)
    print(f"Found {len(substitution_pairs)} substitution pairs.")

    # Step 4: Generate Cypher
    cypher_statements = generate_cypher(substitution_pairs)
    print(f"Generated {len(cypher_statements)} Cypher queries.")

    # Step 5: Push to Neo4j
    push_to_neo4j(cypher_statements, neo4j_uri, neo4j_user, neo4j_password)
    print(f"Substitutes successfully pushed to Neo4j.")

if __name__ == "__main__":
    run_pipeline(
        recipe_filepath='/src/data/RecipeNLG_dataset.csv',
        neo4j_uri='bolt://localhost:7687',
        neo4j_user='database',
        neo4j_password='12345678'
    )
