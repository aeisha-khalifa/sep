from data_load.dataloader import DataLoader
from explain_module.util import summarize_trial, remove_reflections, save_results
from explain_module.agents import PredictReflectAgent
from predict_module.merge_peft_adapter import merge_peft_adapter
from predict_module.supervised_finetune import supervised_finetune
from predict_module.train_reward_model import train_reward_model
from predict_module.tuning_lm_with_rl import tuning_lm_with_rl
from transformers import LlamaTokenizer, pipeline
from trl import AutoModelForCausalLMWithValueHead
import os, json


class Exp_Model:
    def __init__(self, args):
        self.args = args
        self.dataloader = DataLoader(args)

    # ------------------------------------------------------------------
    # Part 1: all OpenAI API calls
    # ------------------------------------------------------------------

    def collect_data(self):
        """Summarize tweets, run self-reflective agents on train data,
        run GPT-3.5 directly on test data as a baseline.
        Saves: SFT data JSON, comparison JSONL, GPT-3.5 test results CSV.
        No local GPU needed."""

        skip_sft_write = getattr(self.args, 'skip_sft_write', False)

        if skip_sft_write:
            print(f"--skip_sft_write: preserving existing {self.args.data_path}")
        else:
            # Clear SFT data file so re-runs don't accumulate duplicates
            if os.path.exists(self.args.data_path):
                os.remove(self.args.data_path)
                print(f"Cleared existing SFT data file: {self.args.data_path}")

        # ── Training data collection ──────────────────────────────────
        print("Loading Train Agents...")
        data = self.dataloader.load(flag="train")

        agents = [PredictReflectAgent(row['ticker'], row['summary'], row['target'])
                  for _, row in data.iterrows()]
        for agent, (_, row) in zip(agents, data.iterrows()):
            agent.date = row['date']
        print("Loaded Train Agents.")

        for agent in agents:
            agent.run()
            if agent.is_correct() and not skip_sft_write:
                prompt = agent._build_agent_prompt()
                response = agent.scratchpad.split('Price Movement: ')[-1]
                sample = {"instruction": prompt, "input": "", "output": response}
                with open(self.args.data_path, 'a') as f:
                    f.write(json.dumps(sample) + "\n")

        correct, incorrect = summarize_trial(agents)
        print(f'Finished Trial 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        comparison_data = []
        for trial in range(self.args.num_reflect_trials):
            for agent in [a for a in agents if not a.is_correct()]:
                prev_response = agent.scratchpad.split('Price Movement: ')[-1]
                agent.run()
                if agent.is_correct():
                    prompt = remove_reflections(agent._build_agent_prompt())
                    response = agent.scratchpad.split('Price Movement: ')[-1]
                    comparison_data.append({
                        "user_input": prompt,
                        "completion_a": prev_response,
                        "completion_b": response
                    })
            correct, incorrect = summarize_trial(agents)
            print(f'Finished Trial {trial+1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

        os.makedirs(self.args.datasets_dir, exist_ok=True)
        comparison_data_path = os.path.join(self.args.datasets_dir, "comparison_data.jsonl")
        if comparison_data:
            with open(comparison_data_path, 'w') as f:
                for item in comparison_data:
                    f.write(json.dumps(item) + "\n")
        print(f"Saved {len(comparison_data)} comparison pairs to {comparison_data_path}")

        # ── GPT-3.5 baseline on test data ────────────────────────────
        print("\nRunning GPT-3.5 baseline on test data...")
        test_data = self.dataloader.load(flag="test")

        test_agents = [PredictReflectAgent(row['ticker'], row['summary'], row['target'])
                       for _, row in test_data.iterrows()]
        for agent, (_, row) in zip(test_agents, test_data.iterrows()):
            agent.date = row['date']

        for agent in test_agents:
            agent.run()

        correct, incorrect = summarize_trial(test_agents)
        print(f'GPT-3.5 test: Correct: {len(correct)}, Incorrect: {len(incorrect)}')
        save_results(test_agents, self.args.save_dir + "gpt_")
        print(f'GPT-3.5 results saved to {self.args.save_dir}gpt_results.csv')

    # ------------------------------------------------------------------
    # Part 2: local GPU only — no API calls
    # ------------------------------------------------------------------

    def train_local(self):
        """SFT + reward model + PPO using data saved by collect_data().
        Reads SFT data JSON and comparison JSONL from Drive. Zero API calls."""

        comparison_data_path = os.path.join(self.args.datasets_dir, "comparison_data.jsonl")
        if not os.path.exists(comparison_data_path):
            raise FileNotFoundError(
                f"comparison_data.jsonl not found at {comparison_data_path}. "
                "Run Part 1 first and make sure at least one agent self-corrected."
            )
        self.args.datasets_dir = comparison_data_path

        supervised_finetune(self.args)
        merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)

        train_reward_model(self.args)
        merge_peft_adapter(model_name=self.args.reward_adapter, output_name=self.args.reward_model_name)

        tuning_lm_with_rl(self.args)
        merge_peft_adapter(model_name=self.args.output_dir + "step_saved",
                           output_name=self.args.sep_model_path)

    # ------------------------------------------------------------------
    # Test: fine-tuned Vicuna on test data
    # ------------------------------------------------------------------

    def test(self):
        print("Loading Test Agents...")
        data = self.dataloader.load(flag="test")

        test_agents = [PredictReflectAgent(row['ticker'], row['summary'], row['target'])
                       for _, row in data.iterrows()]
        for agent, (_, row) in zip(test_agents, data.iterrows()):
            agent.date = row['date']
        print("Loaded Test Agents.")

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.args.sep_model_path,
            load_in_4bit=True,
            device_map="auto"
        )
        tokenizer = LlamaTokenizer.from_pretrained(self.args.output_dir + "step_saved")
        reward_model = pipeline(
            "sentiment-analysis",
            model=self.args.reward_model_name,
            device_map="auto",
            model_kwargs={"load_in_4bit": True},
            tokenizer=tokenizer
        )

        for agent in test_agents:
            agent.run_n_shots(
                model=model,
                tokenizer=tokenizer,
                reward_model=reward_model,
                num_shots=self.args.num_shots
            )

        correct, incorrect = summarize_trial(test_agents)
        print(f'Finished evaluation, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
        save_results(test_agents, self.args.save_dir)

    # ------------------------------------------------------------------
    # Legacy: run everything in one shot (original behaviour)
    # ------------------------------------------------------------------

    def train(self):
        self.collect_data()
        self.train_local()
