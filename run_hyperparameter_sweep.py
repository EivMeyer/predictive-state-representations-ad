import wandb
from omegaconf import DictConfig, OmegaConf
import subprocess
import sys
from pathlib import Path
import argparse
import json
import os
import time
import psutil
import shutil
import torch
import logging
from datetime import datetime
from utils.config_utils import config_wrapper
from utils.training_utils import init_wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sweep.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_gpu_memory_info():
    """Get GPU memory usage information."""
    try:
        return torch.cuda.memory_summary()
    except:
        return "GPU information not available"

def get_system_info():
    """Get system resource information."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent,
        'disk_free_gb': disk.free / (1024**3)
    }

def check_resources():
    """Check if system resources are sufficient to continue."""
    system_info = get_system_info()
    
    # Define thresholds
    DISK_MIN_GB = 10.0
    MEMORY_MAX_PERCENT = 95.0
    
    if system_info['disk_free_gb'] < DISK_MIN_GB:
        raise ResourceWarning(f"Low disk space: {system_info['disk_free_gb']:.1f}GB remaining")
    
    if system_info['memory_percent'] > MEMORY_MAX_PERCENT:
        raise ResourceWarning(f"High memory usage: {system_info['memory_percent']}%")

class SweepCheckpoint:
    def __init__(self, project_name: str):
        self.checkpoint_dir = Path("sweep_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / f"{project_name}_checkpoint.json"
        
    def save(self, sweep_id: str, completed_runs: int):
        # First save to temporary file
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        checkpoint = {
            'sweep_id': sweep_id,
            'completed_runs': completed_runs,
            'timestamp': time.time()
        }
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f)
            
        # Create backup
        backup_file = self.backup_dir / f"{sweep_id}_{completed_runs}_{int(time.time())}.json"
        shutil.copy2(temp_file, backup_file)
        
        # Atomic rename
        temp_file.rename(self.checkpoint_file)
            
    def load(self) -> dict:
        if not self.checkpoint_file.exists():
            # Try to recover from latest backup
            backups = list(self.backup_dir.glob("*.json"))
            if not backups:
                return None
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            shutil.copy2(latest_backup, self.checkpoint_file)
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except:
            return None
            
    def delete(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

def cleanup_between_runs():
    """Cleanup between runs to prevent memory leaks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def train_sweep(total_timesteps, num_envs, completed_runs, checkpoint):
    """Main training function for each sweep run."""
    run_start_time = time.time()
    
    try:
        # Initialize wandb with sweep config and sync_tensorboard=True
        run = wandb.init(sync_tensorboard=True)
        
        # Get the sweep configuration
        sweep_config = wandb.config
        
        # Check system resources
        check_resources()
        
        # Convert sweep config to command line arguments
        cmd_args = []
        for key, value in sweep_config.items():
            cmd_args.append(f"{key}={value}")
        
        # Add total timesteps to command line arguments
        cmd_args.append(f"rl_training.total_timesteps={total_timesteps}")

        # Add num_envs to command line arguments
        cmd_args.append(f"rl_training.num_envs={num_envs}")

        # Disable video logging by setting eval_freq to 0
        cmd_args.append("rl_training.eval_freq=0")

        # Disable checkpointing by setting save_freq to 0
        cmd_args.append("rl_training.save_freq=0")
        
        # Setup logging for this run
        log_dir = Path("logs") / f"run_{wandb.run.id}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"

        command = ["python", "train_rl_agent.py"] + cmd_args

        print(f"Starting run {wandb.run.id} with command: {' '.join(command)}")
        
        # Run train_rl_agent.py with the sweep configuration
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                command,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True
            )
            _, stderr = process.communicate()
            
        # Check for errors
        if process.returncode != 0:
            raise Exception(f"Training process failed:\n{stderr}")
            
        # Log system metrics
        system_info = get_system_info()
        wandb.log({
            "system/cpu_percent": system_info['cpu_percent'],
            "system/memory_percent": system_info['memory_percent'],
            "system/disk_percent": system_info['disk_percent'],
            "system/run_time_hours": (time.time() - run_start_time) / 3600
        })
        
        if torch.cuda.is_available():
            wandb.log({"system/gpu_memory": get_gpu_memory_info()})
            
        # Update checkpoint on successful completion
        completed_runs.append(wandb.run.id)
        checkpoint.save(wandb.run.sweep_id, len(completed_runs))
        
        # Cleanup
        cleanup_between_runs()
        
    except Exception as e:
        logging.error(f"Run failed: {str(e)}")
        wandb.alert(
            title="Training Failed",
            text=f"Error in training process:\n{str(e)}"
        )
        
        # Try to finish wandb run gracefully
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except:
            pass
            
        raise e
    
def define_sweep_configuration():
    """Define the hyperparameter search space and sweep configuration."""
    sweep_config = {
        'method': 'bayes',  # Use Bayesian optimization
        'metric': {
            'name': 'eval/mean_reward',  # Metric to optimize
            'goal': 'maximize'
        },
        'parameters': {
            'rl_training.learning_rate': {
                'distribution': 'log_uniform',
                'min': -11.5129,  # log(1e-5) in base e
                'max': -6.9078    # log(1e-3) in base e
            },
            'rl_training.n_steps': {
                'values': [128, 256, 512, 1024]
            },
            'rl_training.batch_size': {
                'values': [32, 64, 128, 256]
            },
            'rl_training.n_epochs': {
                'values': [3, 5, 8, 10]
            },
            'rl_training.gamma': {
                'distribution': 'uniform',
                'min': 0.95,
                'max': 0.999
            },
            'rl_training.gae_lambda': {
                'distribution': 'uniform',
                'min': 0.9,
                'max': 1.0
            },
            'rl_training.clip_range': {
                'values': [0.1, 0.2, 0.3]
            },
            'rl_training.clip_range_vf': {
                'values': [0.1, 0.2, 0.3]
            },
            'rl_training.ent_coef': {
                'distribution': 'log_uniform',
                'min': -18.4207,  # log(1e-8) in base e
                'max': -2.3026    # log(1e-1) in base e
            },
            'rl_training.vf_coef': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'rl_training.max_grad_norm': {
                'values': [0.3, 0.5, 0.7, 1.0]
            },
            'rl_training.net_arch.pi': {
                'values': [
                    [64, 64],
                    [128, 128],
                    [256, 256],
                    [512, 512],
                    [256, 128, 64],
                    [128, 256, 128]
                ]
            },
            'rl_training.net_arch.vf': {
                'values': [
                    [64, 64],
                    [128, 128],
                    [256, 256],
                    [512, 512],
                    [256, 128, 64],
                    [128, 256, 128]
                ]
            },
            'rl_training.log_std_init': {
                'distribution': 'uniform',
                'min': -3.0,
                'max': 0.0
            },
            'rl_training.minibatch_size': {
                'values': [32, 64, 128]
            },
            'rl_training.use_expln': {
                'values': [True, False]
            },
            'rl_training.full_std': {
                'values': [True, False]
            }
        }
    }
    return sweep_config

@config_wrapper()
def main(cfg: DictConfig) -> None:
    cfg.wandb.enabled = True # Enable wandb for sweep

    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for RL training")
    parser.add_argument('--count', type=int, default=50, help='Number of sweep runs to perform')
    parser.add_argument('--timesteps', type=int, default=100000, 
                       help='Number of timesteps to train each run (default: 100000)')
    parser.add_argument('--num-envs', type=int, default=8, 
                       help='Number of parallel environments to use for training (default 8)')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from last checkpoint if available')
    parser.add_argument('--max-disk-usage', type=float, default=95.0,
                       help='Maximum disk usage percentage before stopping (default: 95.0)')
    args = parser.parse_args()

    # Setup logging directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Initialize checkpoint system
    checkpoint = SweepCheckpoint(cfg.environment)
    completed_runs = []
    
    # Check for existing sweep
    existing_sweep = None
    if args.resume:
        existing_sweep = checkpoint.load()
        if existing_sweep:
            logging.info(f"Found existing sweep checkpoint:")
            logging.info(f"- Sweep ID: {existing_sweep['sweep_id']}")
            logging.info(f"- Completed runs: {existing_sweep['completed_runs']}")
            logging.info(f"- Last updated: {time.ctime(existing_sweep['timestamp'])}")
            
            user_input = input("Do you want to resume this sweep? [y/N]: ")
            if user_input.lower() != 'y':
                existing_sweep = None
                checkpoint.delete()
    
    if not existing_sweep:
        # Check system resources before starting
        system_info = get_system_info()
        logging.info(f"Starting new environment sweep:")
        logging.info(f"- Environment: {cfg.environment}")
        logging.info(f"- Number of runs: {args.count}")
        logging.info(f"- Timesteps per run: {args.timesteps}")
        logging.info(f"- Number of parallel environments: {args.num_envs}")
        logging.info(f"- System Status:")
        logging.info(f"  - CPU Usage: {system_info['cpu_percent']}%")
        logging.info(f"  - Memory Usage: {system_info['memory_percent']}%")
        logging.info(f"  - Disk Usage: {system_info['disk_percent']}%")
        logging.info(f"  - Free Disk Space: {system_info['disk_free_gb']:.1f}GB")
        if torch.cuda.is_available():
            logging.info(f"  - GPU Status:\n{get_gpu_memory_info()}")
        
        # Initialize wandb sweep
        sweep_config = define_sweep_configuration()
        sweep_id = wandb.sweep(sweep_config, project=cfg.environment)
    else:
        sweep_id = existing_sweep['sweep_id']
        completed_runs = existing_sweep['completed_runs']
        remaining_runs = args.count - len(completed_runs)
        logging.info(f"Resuming sweep with {remaining_runs} remaining runs")
    
    # Start the sweep
    try:
        wandb.agent(
            sweep_id, 
            lambda: train_sweep(args.timesteps, args.num_envs, completed_runs, checkpoint),
            count=args.count - len(completed_runs)
        )
        logging.info(f"Sweep completed successfully!")
        checkpoint.delete()
        
    except KeyboardInterrupt:
        logging.info("\nSweep interrupted! Progress has been saved.")
        logging.info("To resume, run the same command with --resume flag")
        sys.exit(1)
    except ResourceWarning as e:
        logging.error(f"Resource warning: {str(e)}")
        logging.info("Sweep stopped due to resource constraints. Progress saved.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.info("Sweep failed but progress was saved. Can resume with --resume flag")
        sys.exit(1)

if __name__ == "__main__":
    main()