import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from score_network import train_score_network, compute_score

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Score Matching) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(returns) < win + 10:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            ret_win = returns.iloc[-win:].values
            if ret_win.shape[0] < 50:
                continue
            # Train score network
            input_dim = ret_win.shape[1]
            model = train_score_network(ret_win, input_dim,
                                        hidden_dim=config.HIDDEN_DIM,
                                        n_layers=config.N_LAYERS,
                                        noise_scale=config.NOISE_SCALE,
                                        lr=config.LEARNING_RATE,
                                        batch_size=config.BATCH_SIZE,
                                        epochs=config.EPOCHS)
            # Compute score for the most recent observation (last day)
            last_obs = ret_win[-1:].reshape(1, -1)
            score = compute_score(model, last_obs)[0]  # shape (input_dim,)
            # Score can be positive or negative. We'll use the absolute value as signal strength,
            # but the sign indicates direction (positive = upward momentum)
            scores = {tickers[i]: score[i] for i in range(input_dim)}
            window_results[win] = scores
            for etf, sc in scores.items():
                # Use absolute score as instability/momentum strength
                strength = abs(sc)
                if etf not in best_per_etf or strength > best_per_etf[etf][0]:
                    best_per_etf[etf] = (strength, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": float(score), "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "score_magnitude": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by score magnitude: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/score_match_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Score Matching Diffusion Sampler complete ===")

if __name__ == "__main__":
    main()
