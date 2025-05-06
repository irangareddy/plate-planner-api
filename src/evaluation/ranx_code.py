
import pandas as pd
from ranx import Qrels, Run, evaluate


# Simulated ground-truth and prediction generation based on example queries
def build_test_queries() -> list[tuple[list[str], str]]:
    return [
        (["butter", "sugar", "flour"], "Scottish Shortbread"),
        (["lettuce", "tomato", "olive oil"], "Fat Free Salad With Fiber"),
        (["pasta", "garlic", "parmesan"], "Janet'S Pasta"),
        (["chicken", "carrot", "onion"], "Savory Chicken Roll-Ups"),
        (["egg", "milk", "banana"], "Banana Egg Nog"),
    ]

def build_qrels_and_run(test_queries, simulated_results):
    qrels_dict = {}
    run_dict = {}

    for i, (ingredients, expected_title) in enumerate(test_queries):
        qid = f"q{i}"
        qrels_dict[qid] = {expected_title: 1}

        predicted = simulated_results[i]
        run_dict[qid] = {item["title"]: item["combined_score"] for item in predicted}

    return Qrels(qrels_dict), Run(run_dict)

# Simulated model output per query (truncated example based on user input)
simulated_results = [
    [{"title": "Scottish Shortbread", "combined_score": 1.0},
     {"title": "Rhubarb Dessert #2", "combined_score": 1.0},
     {"title": "Scotch Shortbread", "combined_score": 1.0},
     {"title": "Butter Cookies", "combined_score": 1.0},
     {"title": "Scottish Shortbread", "combined_score": 1.0}],

    [{"title": "Stir-Fried Tomatoes, Pepper And Onions", "combined_score": 0.7123},
     {"title": "Fat Free Salad With Fiber", "combined_score": 0.7014},
     {"title": "Chicken breasts with thyme and vegetables", "combined_score": 0.6958}],

    [{"title": "Janet'S Pasta", "combined_score": 0.7202},
     {"title": "One Pot Fettucine Alfredo", "combined_score": 0.7183},
     {"title": "Spaghetti carbonara (5 ingredient )", "combined_score": 0.7170}],

    [{"title": "Savory Chicken Roll-Ups", "combined_score": 0.9545},
     {"title": "Yellow Split Pea Soup", "combined_score": 0.9417},
     {"title": "Grandma Brady'S Chicken", "combined_score": 0.9311}],

    [{"title": "Banana Egg Nog", "combined_score": 0.9889},
     {"title": "The Easiest Banana Bread", "combined_score": 0.9556},
     {"title": "Chocolate Banana High Protein Diet Shake", "combined_score": 0.9389}],
]

def main():
    test_queries = build_test_queries()
    qrels, run = build_qrels_and_run(test_queries, simulated_results)

    # Evaluate metrics
    metrics = ["precision@5", "recall@5", "ndcg@5", "mrr"]
    results = evaluate(qrels=qrels, run=run, metrics=metrics)

    # Display results as DataFrame
    df = pd.DataFrame([results])
    print("Evaluation Metrics:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
