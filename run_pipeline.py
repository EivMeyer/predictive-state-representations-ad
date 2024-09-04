import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.errors import ConfigCompositionException

# Assume your main config is named 'config.yaml' and located in the current directory
CONFIG_PATH = "config.yaml"

def load_base_config():
    return OmegaConf.load(CONFIG_PATH)

def validate_config_overrides(config_overrides):
    base_config = load_base_config()
    try:
        # Create a new config with the overrides
        cli_conf = OmegaConf.from_dotlist(config_overrides)
        # Merge the base config with the overrides
        merged_conf = OmegaConf.merge(base_config, cli_conf)
        # If we get here, the config is valid
        return True, merged_conf
    except ConfigCompositionException as e:
        return False, str(e)
    

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

The script will guide you through the process, asking for necessary inputs along the way.
You can modify any configuration parameter by passing it as an argument.
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

def select_model() -> Optional[str]:
    models_dir = "./output/models"
    model_files = sorted(
        [os.path.join(dp, f) for dp, dn, filenames in os.walk(models_dir) for f in filenames if f.endswith('.pth')],
        key=os.path.getmtime, reverse=True
    )[:20]
    
    if not model_files:
        print("No existing model files found.")
        return None

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

def show_system_info():
    print("System Information:")
    print("CPU Info:")
    os.system("lscpu | grep 'Model name\|CPU(s):'")
    print("Memory Info:")
    os.system("free -h")
    print("GPU Info:")
    if subprocess.call("which nvidia-smi", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        os.system("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader")
    else:
        print("NVIDIA GPU not detected or nvidia-smi not available")

def run_pipeline(config: Dict[str, Any]):
    if not config['use_existing_dataset']:
        print("\nStep 1: Collecting new dataset...")
        cmd = [
            "./parallel_dataset_collection.sh",
            "-w", str(config['num_workers']),
            "-e", str(config['total_episodes']),
            "-r", str(config['episodes_per_restart'])
        ] + config['config_overrides']
        subprocess.run(cmd, check=True)

    if config['train_new_model']:
        print("\nStep 2: Training representation model...")
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
        print("\nStep 2: Using existing representation model...")

    print(f"Using representation model: {config['model_path']}")

    print("\nStep 3: Training RL model...")
    cmd = [
        "python", "train_rl_agent.py",
        f"rl_training.total_timesteps={config['rl_timesteps']}",
        f"rl_training.num_envs={config['rl_num_envs']}",
        f"representation.model_path={config['model_path']}"
    ] + config['config_overrides']
    subprocess.run(cmd, check=True)

    print("\nML pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Script")
    parser.add_argument("--nohup", action="store_true", help="Run with nohup")
    parser.add_argument("--output", help="Specify the output file")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode")
    parser.add_argument("config_overrides", nargs="*", help="Configuration overrides")
    args = parser.parse_args()

    show_readme()
    show_system_info()

    config = {
        'use_existing_dataset': False,
        'num_workers': 16,
        'total_episodes': 10000,
        'episodes_per_restart': 500,
        'train_new_model': True,
        'representation_epochs': 1000,
        'model_path': "",
        'rl_timesteps': 10000000,
        'rl_num_envs': 16,
        'config_overrides': args.config_overrides
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
    for key, value in config.items():
        if key != 'config_overrides':
            print(f"- {key}: {value}")
    print(f"- config_overrides: {' '.join(config['config_overrides'])}")
    print(f"- output file: {args.output or 'pipeline_output.log'}")
    print(f"- running with nohup: {args.nohup}")

    if not args.non_interactive:
        print()
        if input("Do you want to proceed with this configuration? (y/n) ").lower() != 'y':
            print("Aborted by user.")
            sys.exit(0)

    cmd = f"python {__file__} --non-interactive " + " ".join(sys.argv[1:])
    if args.nohup:
        output_file = args.output or f"ml_pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd = f"nohup {cmd} > {output_file} 2>&1 &"
        print(f"Starting pipeline with nohup. Output will be written to {output_file}")
        os.system(cmd)
        print(f"Pipeline started in background. Use 'tail -f {output_file}' to monitor progress.")
    else:
        run_pipeline(config)

if __name__ == "__main__":
    main()