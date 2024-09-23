import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler

class NoScheduler(_LRScheduler):
    def __init__(self, optimizer):
        super(NoScheduler, self).__init__(optimizer)

    def step(self):
        pass

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def init_wandb(cfg: DictConfig):
    if cfg.wandb.enabled:
        import wandb
        wandb.init(project=cfg.wandb.project + "-" + cfg.environment, config=OmegaConf.to_container(cfg, resolve=True))
        print(f"Initialized wandb project: {cfg.wandb.project}")
        return wandb
    else:
        # Return a dummy object that does nothing when called
        class DummyWandB:
            def __getattr__(self, _):
                return lambda *args, **kwargs: None
        print("WandB is disabled")
        return DummyWandB()
    