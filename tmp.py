import json

with open("data/ropes/ropes_no_coref_dev_combined_examples.json", 'r') as f:
    data = json.load(f)
    for article in data:
        if "fact_labels" in article:
            print(article["background"])
            print(article["situation"])
            print([article["context_sent"][i] for i in article["fact_labels"] if i != len(article["context_sent"])])
            print(article["question"])
            print(article["answer"])
            print()