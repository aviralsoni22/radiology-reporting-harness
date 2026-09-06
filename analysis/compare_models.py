import csv
import random
import sys
import concurrent.futures

sys.path.insert(0, "src")
from pipeline import build_report, make_client, MODELS
from retrieve import Retriever
from scorer import res_case

def run_bakeoff():
    print("Loading data...")
    rows = list(csv.DictReader(open("data/train.csv", encoding="utf-8-sig")))
    random.seed(42)
    random.shuffle(rows)
    
    # Use a small holdout set for the bakeoff to save time/cost
    holdout = rows[:20]
    fit = rows[20:]
    
    R = Retriever(fit, k=3)
    
    # Models you might want to compare
    models_to_test = ["echo", "GLM"] 
    
    results = {}
    
    for model_key in models_to_test:
        print(f"\n--- Testing model: {model_key} ---")
        client = make_client(model_key)
        
        # 1. Generate all hypotheses in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            hyps = list(executor.map(lambda r: build_report(r, client, retriever=R), holdout))
            
        # 2. Score sequentially
        scored = []
        for i, (row, hyp) in enumerate(zip(holdout, hyps), 1):
            res, F, I = res_case(row["template_content"], row["report"], hyp)
            scored.append(res)
            
            if i % 5 == 0:
                print(f"  Processed {i}/{len(holdout)} cases. Current mean RES: {sum(scored)/len(scored):.4f}")
                
        final_res = sum(scored)/len(scored)
        results[model_key] = final_res
        print(f"-> FINAL LOCAL RES FOR {model_key.upper()}: {final_res:.4f}")
        
        # Save output to file in submission format
        out_path = f"analysis/compare_results_{model_key}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["case_id", "report"])
            for row, hyp in zip(holdout, hyps):
                writer.writerow([row["case_id"], hyp])
        print(f"-> Saved submission-formatted output to {out_path}\n")

    print("\n================ FINAL LEADERBOARD ================")
    for m, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{m.ljust(15)} : {score:.4f} RES (lower is better)")

if __name__ == "__main__":
    run_bakeoff()