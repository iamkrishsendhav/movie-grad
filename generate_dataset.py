import os
import numpy as np
import pandas as pd

def main(path="data/movies_100k.csv", n_rows=100_000):
    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(42)

    genres = ['Action','Comedy','Drama','Thriller','Horror','Romance','Sci-Fi','Fantasy']

    budget = np.round(rng.normal(50, 30, n_rows).clip(1, 400), 2)
    director_score = np.round(rng.normal(6, 2, n_rows).clip(0,10), 2)
    cast_popularity = np.round(rng.normal(50, 20, n_rows).clip(0,100), 2)
    runtime = rng.integers(80, 181, n_rows)
    release_month = rng.integers(1, 13, n_rows)
    marketing_spend = np.round(rng.normal(10, 5, n_rows).clip(0,100), 2)
    is_sequel = rng.integers(0,2,n_rows)
    genre = rng.choice(genres, n_rows)

    prob = (budget*0.004 + director_score*0.1 +
            cast_popularity*0.01 + marketing_spend*0.02 +
            is_sequel*0.05 + (runtime-100)*0.001)

    prob = 1 / (1 + np.exp(- (prob - np.mean(prob)) / np.std(prob)))

    df = pd.DataFrame({
        "genre": genre,
        "budget_million": budget,
        "director_score": director_score,
        "cast_popularity": cast_popularity,
        "runtime_minutes": runtime,
        "release_month": release_month,
        "marketing_spend": marketing_spend,
        "is_sequel": is_sequel,
        "box_office_hit": (prob > 0.55).astype(int)
    })

    df.to_csv(path, index=False)
    print("Dataset saved to:", path)

if __name__ == "__main__":
    main()
