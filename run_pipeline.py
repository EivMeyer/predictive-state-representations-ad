import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any

def show_readme():
    print("""
ML Pipeline Script
==================

This script automates the process of dataset collection, representation model training,
and reinforcement learning model training.

Usage:
    python run_pipeline.py [OPTIONS] [CONFIG_OVERRIDES]

Options:
    --nohup             Run the script with nohup (keeps running after you log out)
    --output FILE       Specify the output file (default: ml_pipeline_output_TIMESTAMP.log)
    --non-interactive   Run in non-interactive mode (requires all parameters to be specified)

Config Overrides:
    key=value           Override any configuration parameter (e.g., training.learning_rate=64)

Examples:
    python run_pipeline.py
    python run_pipeline.py --nohup --output my_run.log
    python run_pipeline.py training.learning_rate=64 dataset.t_pred=50 viewer.window_size=256
""")

def get_input(prompt: str, default: str) -> str:
    return input(f"{prompt} [{default}]: ").strip() or default

def check_dataset() -> bool:
    dataset_dir = "./output/dataset"
    if os.path.isdir(dataset_dir):
        file_count = len([f for f in os.listdir(dataset_dir) if f.startswith("batch_") and f.endswith(".pt")])
        total_size = subprocess.check_output(['du', '-sh', dataset_dir]).split()[0].decode('utf-8')
        last_modified = datetime.fromtimestamp(os.path.getmtime(dataset_dir)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Existing dataset found:")
        print(f"- Number of batch files: {file_count}")
        print(f"- Total size: {total_size}")
        print(f"- Last modified: {last_modified}")
        return input("Do you want to use this existing dataset? (y/n) ").lower() == 'y'
    return False

def select_model() -> str:
    models_dir = "./output/models"
    model_files = sorted(
        [os.path.join(dp, f) for dp, dn, filenames in os.walk(models_dir) for f in filenames if f.endswith('.pth')],
        key=os.path.getmtime, reverse=True
    )[:20]
    
    if not model_files:
        print("No existing model files found.")
        return ""

    print("Select a model file:")
    for i, model_path in enumerate(model_files, 1):
        print(f"{i}. {model_path}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(model_files):
                return model_files[choice - 1]
        except ValueError:
            pass
        print("Invalid selection. Please try again.")

def generate_reproducible_command(config: Dict[str, Any], args: argparse.Namespace) -> str:
    cmd_parts = ["python3", "run_pipeline.py", "--non-interactive"]
    
    if args.nohup:
        cmd_parts.append("--nohup")
    
    if args.output:
        cmd_parts.extend(["--output", args.output])
    
    # Include pipeline control parameters
    cmd_parts.extend([
        f"--collect_new_dataset={'true' if not config['use_existing_dataset'] else 'false'}",
        f"--train_new_model={'true' if config['train_new_model'] else 'false'}",
    ])
    
    # Include configuration parameters
    cmd_parts.extend([
        f"dataset.num_workers={config['num_workers']}",
        f"dataset.num_episodes={config['total_episodes']}",
        f"training.epochs={config['representation_epochs']}",
        f"representation.model_path={config['model_path']}",
        f"rl_training.total_timesteps={config['rl_timesteps']}",
        f"rl_training.num_envs={config['rl_num_envs']}",
    ])
    
    # Add any additional config overrides
    cmd_parts.extend(config['config_overrides'])
    
    return " ".join(cmd_parts)

def run_pipeline(config: Dict[str, Any]):
    if not config['use_existing_dataset']:
        print("Step 1: Collecting new dataset...")
        cmd = [
            "./parallel_dataset_collection.sh",
            "-w", str(config['num_workers']),
            "-e", str(config['total_episodes']),
            "-r", str(config['episodes_per_restart'])
        ] + config['config_overrides']
        subprocess.run(cmd, check=True)
    else:
        print("Step 1: Using existing dataset.")

    if config['train_new_model']:
        print("Step 2: Training representation model...")
        cmd = [
            "python", "train_model.py",
            f"training.epochs={config['representation_epochs']}"
        ] + config['config_overrides']
        subprocess.run(cmd, check=True)

        latest_model = max(
            (os.path.join(root, name) for root, _, files in os.walk("./output/models") for name in files if name == "final_model.pth"),
            key=os.path.getmtime
        )
        config['model_path'] = os.path.relpath(latest_model)
    else:
        print("Step 2: Using existing representation model...")

    print(f"Using representation model: {config['model_path']}")

    print("Step 3: Training RL model...")
    cmd = [
        "python", "train_rl_agent.py",
        f"rl_training.total_timesteps={config['rl_timesteps']}",
        f"rl_training.num_envs={config['rl_num_envs']}",
        f"representation.model_path={config['model_path']}"
    ] + config['config_overrides']
    subprocess.run(cmd, check=True)

    print("ML pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Script")
    parser.add_argument("--nohup", action="store_true", help="Run with nohup")
    parser.add_argument("--output", help="Specify the output file")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("--collect_new_dataset", type=str, choices=['true', 'false'], help="Whether to collect a new dataset")
    parser.add_argument("--train_new_model", type=str, choices=['true', 'false'], help="Whether to train a new representation model")
    parser.add_argument("config_overrides", nargs="*", help="Configuration overrides")
    args = parser.parse_args()

    show_readme()

    config = {
        'use_existing_dataset': not (args.collect_new_dataset == 'true' if args.collect_new_dataset else False),
        'num_workers': 16,
        'total_episodes': 10000,
        'episodes_per_restart': 500,
        'train_new_model': args.train_new_model == 'true' if args.train_new_model else True,
        'representation_epochs': 1000,
        'model_path': "",
        'rl_timesteps': 10000000,
        'rl_num_envs': 16,
        'config_overrides': [override for override in args.config_overrides if not override.startswith(("collect_new_dataset=", "train_new_model="))]
    }

    if not args.non_interactive:
        config['use_existing_dataset'] = check_dataset()
        if not config['use_existing_dataset']:
            config['num_workers'] = int(get_input("Enter the number of workers for dataset collection", str(config['num_workers'])))
            config['total_episodes'] = int(get_input("Enter the total number of episodes to collect", str(config['total_episodes'])))
            config['episodes_per_restart'] = int(get_input("Enter the number of episodes to collect before restarting a worker", str(config['episodes_per_restart'])))

        config['train_new_model'] = input("Do you want to train a new representation model? (y/n) ").lower() == 'y'
        if config['train_new_model']:
            config['representation_epochs'] = int(get_input("Enter the number of epochs for training the representation model", str(config['representation_epochs'])))
        else:
            config['model_path'] = select_model()
            if not config['model_path']:
                print("Error: No existing model selected and not training a new one. Cannot proceed.")
                sys.exit(1)

        config['rl_timesteps'] = int(get_input("Enter the total number of timesteps for RL training", str(config['rl_timesteps'])))
        config['rl_num_envs'] = int(get_input("Enter the number of parallel environments for RL training", str(config['rl_num_envs'])))

    print("\nConfiguration:")
    print(f"- collect_new_dataset: {not config['use_existing_dataset']}")
    print(f"- num_workers: {config['num_workers']}")
    print(f"- total_episodes: {config['total_episodes']}")
    print(f"- episodes_per_restart: {config['episodes_per_restart']}")
    print(f"- train_new_model: {config['train_new_model']}")
    print(f"- representation_epochs: {config['representation_epochs']}")
    print(f"- model_path: {config['model_path']}")
    print(f"- rl_timesteps: {config['rl_timesteps']}")
    print(f"- rl_num_envs: {config['rl_num_envs']}")
    print(f"- config_overrides: {' '.join(config['config_overrides'])}")
    print(f"- output file: {args.output or 'pipeline_output.log'}")
    print(f"- running with nohup: {args.nohup}")

    # Generate and display the reproducible command
    reproducible_cmd = generate_reproducible_command(config, args)
    print("\nEquivalent one-liner command to reproduce this run:")
    print(reproducible_cmd)

    if not args.non_interactive:
        if input("\nDo you want to proceed with this configuration? (y/n) ").lower() != 'y':
            print("Aborted by user.")
            sys.exit(0)

    if args.nohup:
        output_file = args.output or f"ml_pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        nohup_cmd = f"nohup {reproducible_cmd} > {output_file} 2>&1 &"
        print(f"Starting pipeline with nohup. Output will be written to {output_file}")
        os.system(nohup_cmd)
        print(f"Pipeline started in background. Use 'tail -f {output_file}' to monitor progress.")
    else:
        run_pipeline(config)

if __name__ == "__main__":
    main()