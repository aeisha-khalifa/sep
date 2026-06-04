"""
Run one experimental condition and save results to CSV.

Conditions:
  A  --  tweets only      (baseline, replicates paper's GPT-3.5 row)
  B  --  tweets + indicators
  C  --  indicators only
"""
import argparse
import os
import json
from dotenv import load_dotenv
load_dotenv()
from explain_module.util import remove_fewshot

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str, required=True, choices=["A", "B", "C"])
parser.add_argument("--price_dir",  type=str, default="data/price/preprocessed/")
parser.add_argument("--tweet_dir",  type=str, default="C:/Users/aeish/sn2/tweet/raw/")
parser.add_argument("--ohlcv_dir",  type=str, default="data/ohlcv/")
parser.add_argument("--cache_path", type=str, default="data/summary_cache.pkl")
parser.add_argument("--seq_len",    type=int, default=5)
parser.add_argument("--num_reflect_trials", type=int, default=3)
parser.add_argument("--max_samples", type=int, default=0,
                    help="Limit samples for testing (0 = no limit)")
parser.add_argument("--results_dir", type=str, default="results/")
args = parser.parse_args()

# Set condition flags
args.use_indicators   = args.condition in ("B",)
args.indicators_only  = args.condition == "C"

print(f"Condition {args.condition}: use_indicators={args.use_indicators}, indicators_only={args.indicators_only}")

from data_load.dataloader import DataLoader
from explain_module.agents import PredictReflectAgent
from explain_module.util import save_results, summarize_trial

# Load data
print("Loading data...")
dataloader = DataLoader(args)
data = dataloader.load(flag="test")
print(f"Loaded {len(data)} samples.")

if args.max_samples > 0:
    data = data.head(args.max_samples)
    print(f"Limited to {len(data)} samples for testing.")

# Build agents and attach date for downstream training data conversion
agents = []
for _, row in data.iterrows():
    agent = PredictReflectAgent(row["ticker"], row["summary"], row["target"])
    agent.date = row["date"]
    agents.append(agent)

# Run reflection loop — also capture (wrong → correct) pairs for reward model training
reflection_pairs = []

for i, agent in enumerate(agents):
    print(f"[{i+1}/{len(agents)}] {agent.ticker} | target={agent.target}")
    agent.run()

    for _ in range(args.num_reflect_trials - 1):
        if agent.is_correct():
            break
        # Capture the wrong response BEFORE the next reflection attempt
        wrong_response = agent.scratchpad.split('Price Movement: ')[-1].strip()
        agent.run(reset=False)

        if agent.is_correct():
            # Capture the correct response AFTER the reflection changed the prediction
            correct_response = agent.scratchpad.split('Price Movement: ')[-1].strip()
            reflection_pairs.append({
                "user_input":   remove_fewshot(agent._build_agent_prompt()),
                "completion_a": wrong_response,    # incorrect prediction
                "completion_b": correct_response,  # correct prediction after reflection
                "ticker":       agent.ticker,
                "date":         agent.date,
                "target":       agent.target
            })

correct, incorrect = summarize_trial(agents)
print(f"\nCondition {args.condition} — Correct: {len(correct)} / {len(agents)} "
      f"({100*len(correct)/len(agents):.1f}%)")

# Save results CSV
os.makedirs(args.results_dir, exist_ok=True)
save_results(agents, os.path.join(args.results_dir, f"condition_{args.condition}_"))
print(f"Results saved to {args.results_dir}condition_{args.condition}_results.csv")

# Save reflection pairs (needed for reward model training for this condition)
pairs_path = os.path.join(args.results_dir, f"condition_{args.condition}_reflection_pairs.json")
with open(pairs_path, "w") as f:
    json.dump(reflection_pairs, f)
print(f"Reflection pairs saved: {len(reflection_pairs)} pairs -> {pairs_path}")
