import requests
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============ CONFIG ============
TMDB_KEY = "e6d494ee22319451c86b835d1d9f81ec"       
TOTAL_MOVIES = 5000
OUTPUT_FILE = "movies_data.json"
MAX_WORKERS = 15                 # Number of concurrent downloads (safe for TMDb limits)
# ================================

PROGRESS_FILE = "scrape_progress.json"


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"collected_ids": [], "failed_ids": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_data(movies):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(movies, f, indent=2)


def get_movie_ids(total):
    """Get movie IDs from popular + top rated pages"""
    all_ids = []
    endpoints = ["movie/popular", "movie/top_rated"]

    for endpoint in endpoints:
        page = 1
        # Each endpoint can safely yield up to 10,000 items (500 pages)
        while len(all_ids) < total and page <= 500:
            try:
                resp = requests.get(
                    f"https://api.themoviedb.org/3/{endpoint}",
                    params={"api_key": TMDB_KEY, "page": page},
                    timeout=10
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])

                if not False:  # Clean extraction
                    for m in results:
                        if m["id"] not in all_ids:
                            all_ids.append(m["id"])
                            if len(all_ids) >= total:
                                break
                page += 1
                time.sleep(0.1)
            except Exception as e:
                print(f"  Error fetching IDs from {endpoint} page {page}: {e}")
                break
    return all_ids[:total]


def get_movie_details(movie_id):
    """Get full details + credits in a SINGLE request using append_to_response"""
    base = "https://api.themoviedb.org/3"
    
    # Combined API request
    response = requests.get(
        f"{base}/movie/{movie_id}",
        params={
            "api_key": TMDB_KEY,
            "append_to_response": "credits"  # Combines details & credits!
        },
        timeout=10
    )
    response.raise_for_status()
    data = response.json()

    credits = data.get("credits", {})
    
    director = next(
        (c["name"] for c in credits.get("crew", []) if c["job"] == "Director"),
        None
    )
    
    writers = [
        c["name"] for c in credits.get("crew", [])
        if c["job"] in ("Writer", "Screenplay", "Story")
    ][:3]

    cast = [
        {"name": c["name"], "character": c["character"]}
        for c in credits.get("cast", [])[:10]
    ]

    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "year": data.get("release_date", "")[:4] if data.get("release_date") else None,
        "release_date": data.get("release_date"),
        "genres": [g["name"] for g in data.get("genres", [])],
        "rating": data.get("vote_average"),
        "director": director,
        "cast": cast,
        "overview": data.get("overview", ""),
        "original_language": data.get("original_language")
    }


def main():
    print("=" * 50)
    print("TMDB Movie Data Collector (Thread-Optimized)")
    print("=" * 50)

    progress = load_progress()
    movies = load_existing_data()
    collected_ids = set(progress["collected_ids"])
    failed_ids = set(progress["failed_ids"])

    print(f"Already collected: {len(movies)}")
    print(f"Failed: {len(failed_ids)}")

    print(f"\nFetching {TOTAL_MOVIES} movie IDs...")
    all_ids = get_movie_ids(TOTAL_MOVIES)
    print(f"Got {len(all_ids)} IDs")

    remaining = [mid for mid in all_ids if mid not in collected_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nDone! Saved to", OUTPUT_FILE)
        return

    print(f"\nCollecting with {MAX_WORKERS} workers parallelly...\n")
    start = time.time()
    
    # Thread pool execution loop
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to threads
        future_to_id = {executor.submit(get_movie_details, mid): mid for mid in remaining if mid not in failed_ids}
        
        for i, future in enumerate(as_completed(future_to_id), 1):
            movie_id = future_to_id[future]
            try:
                movie = future.result()
                movies.append(movie)
                collected_ids.add(movie_id)
                print(f"[{i}/{len(remaining)}] ✓ {movie['title']} ({movie['year']})")
            except Exception as e:
                failed_ids.add(movie_id)
                print(f"[{i}/{len(remaining)}] ✗ ID {movie_id} failed: {e}")

            # Batch save state to disk every 25 movies to optimize file I/O operations
            if i % 25 == 0 or i == len(remaining):
                save_data(movies)
                save_progress({
                    "collected_ids": list(collected_ids),
                    "failed_ids": list(failed_ids)
                })
                elapsed = time.time() - start
                eta = (elapsed / i) * (len(remaining) - i)
                print(f"    ---> Progress Saved | Total Collected: {len(movies)} | ETA: {eta/60:.1f}min\n")

    print("\n" + "=" * 50)
    print(f"Done! {len(movies)} movies saved to {OUTPUT_FILE}")
    print(f"Failed: {len(failed_ids)}")
    print("=" * 50)


if __name__ == "__main__":
    main()