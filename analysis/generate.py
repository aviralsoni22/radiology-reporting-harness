import csv
import sys
import concurrent.futures

sys.path.insert(0, "src")
from pipeline import build_report, make_client, write_submission
from retrieve import Retriever

def generate_submission():
    print("Loading datasets...")
    # Using utf-8-sig to safely handle any hidden BOM characters like we saw in train.csv
    train = list(csv.DictReader(open("data/train.csv", encoding="utf-8-sig")))
    test = list(csv.DictReader(open("data/test.csv", encoding="utf-8-sig")))
    
    R = Retriever(train, k=3)
    
    # We will use GLM since it is currently our best model
    client = make_client("GLM")
    print(f"Generating reports for {len(test)} test cases using GLM in parallel...")
    
    # Generate hypotheses in parallel for speed
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # We process all rows in the test set
        hyps = list(executor.map(lambda r: build_report(r, client, retriever=R), test))
        
    for row, hyp in zip(test, hyps):
        results.append({"case_id": row["case_id"], "report": hyp})
        
    # Write the final file using the exact quoting standard Kaggle wants
    write_submission(results, "submission.csv")
    print("Generation complete!")

if __name__ == "__main__":
    generate_submission()
