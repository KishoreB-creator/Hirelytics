import pandas as pd
import requests
from tqdm import tqdm
import time
import difflib
import json

# ---------------- CONFIG ----------------
API_URL = "https://shl-recommender.azurewebsites.net/recommend"
BATCH_SIZE = 10
MAX_REQUESTS_PER_MIN = 15
DELAY_BETWEEN_BATCHES = 60 / MAX_REQUESTS_PER_MIN  
MAX_RETRIES = 3
TIMEOUT = 90  
FUZZY_THRESHOLD = 0.80  

# ---------------- LOAD DATA ----------------
excel = pd.ExcelFile("Gen_AI Dataset.xlsx")
train_df = pd.read_excel(excel, "Train-Set")
test_df = pd.read_excel(excel, "Test-Set")

# Clean NaNs if any
train_df = train_df.dropna(subset=["Query", "Assessment_url"])
test_df = test_df.dropna(subset=["Query"])

# ---------------- URL NORMALIZATION ----------------
def normalize_url(url):
    """Standardize SHL URLs (lowercase, remove trailing slashes)."""
    if not isinstance(url, str):
        return ""
    url = url.strip().lower()
    url = url.replace("https://", "").replace("http://", "")
    return url.strip("/")

# ---------------- FUZZY MATCHING ----------------
def fuzzy_match(url, candidates, threshold=FUZZY_THRESHOLD):
    """Return True if any candidate is a close fuzzy match to url."""
    for cand in candidates:
        ratio = difflib.SequenceMatcher(None, url.lower(), cand.lower()).ratio()
        if ratio >= threshold:
            return True
    return False

# ---------------- API CALL FUNCTION ----------------
def call_api(query, k=10):
    """Call FastAPI backend with retry handling."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.post(API_URL, json={"query": query, "k": k}, timeout=TIMEOUT)
            res.raise_for_status()
            data = res.json()

            
            recs = []
            if "recommendations" in data:
                recs = data.get("recommendations", [])
            elif "recommended_assessments" in data:
                recs = data.get("recommended_assessments", [])
            else:
                print(f"⚠️ Unexpected response format for query: {query[:40]}")
                return []

            
            urls = [normalize_url(r.get("url", r.get("assessment_url", ""))) for r in recs if r]
            return urls

        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed for query: {query[:60]} -> {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3)
            else:
                return []

def get_batch_recommendations(queries, k=10):
    """Process a batch of queries (with delay for Gemini rate limit)."""
    results = []
    for query in queries:
        recs = call_api(query, k)
        results.append((query, recs))
    time.sleep(DELAY_BETWEEN_BATCHES)
    return results

# ---------------- EVALUATION FUNCTION ----------------
def recall_at_k(train_df, k=10):
    """Compute mean Recall@K with grouped queries."""
    grouped = train_df.groupby("Query")["Assessment_url"].apply(list).reset_index()
    scores = []

    print(f"📊 Evaluating {len(grouped)} unique queries...")
    for i in tqdm(range(0, len(grouped), BATCH_SIZE)):
        batch = grouped.iloc[i:i + BATCH_SIZE]
        queries = batch["Query"].tolist()
        batch_results = get_batch_recommendations(queries, k)

        for (query, preds), (_, row) in zip(batch_results, batch.iterrows()):
            true_urls = [normalize_url(u) for u in row["Assessment_url"]]
            if not preds:
                scores.append(0)
                continue

            
            hits = len(set(preds) & set(true_urls))

            
            for true_url in true_urls:
                if true_url not in preds and fuzzy_match(true_url, preds):
                    hits += 1

            recall = hits / len(true_urls)
            scores.append(min(recall, 1.0))  # clip to 1.0 max

    mean_recall = sum(scores) / len(scores) if scores else 0.0
    return mean_recall

# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    print("⏳ Evaluating Mean Recall@10...")
    mean_recall = recall_at_k(train_df, k=10)
    print(f"\n✅ Mean Recall@10 = {mean_recall:.3f}\n")

    # ---------------- GENERATE SUBMISSION ----------------
    print(f"📁 Generating submission for {len(test_df)} test queries...")
    rows = []

    for i in tqdm(range(0, len(test_df), BATCH_SIZE)):
        batch = test_df.iloc[i:i + BATCH_SIZE]
        queries = batch["Query"].tolist()
        batch_results = get_batch_recommendations(queries, k=10)

        for query, preds in batch_results:
            for url in preds:
                if not url:
                    continue
                formatted_url = "https://" + url if not url.startswith("http") else url
                rows.append({"Query": query, "Assessment_url": formatted_url})

    sub_df = pd.DataFrame(rows)
    sub_df.to_csv("Predictions.csv", index=False)
    print(f"\n✅ Saved Predictions.csv with {len(sub_df)} rows at {time.strftime('%H:%M:%S')}")
