from copy import deepcopy

from torch import nn

from torchtrainer.engine.config import Config, instantiate


class CompositeLoss(nn.Module):
    """Container class to combine multiple loss functions."""
    def __init__(self, cfg_loss: Config, return_logs: bool = True, prefix: str = ""):
        """Args:
            cfg_loss: Configuration dictionary for the loss functions.
            return_logs: Whether to return individual loss values as logs.
            prefix: Prefix to add to the loss names in the logs.
        """
        super().__init__()
        self.loss_map = nn.ModuleDict()
        self.cfg_loss = cfg_loss
        self.return_logs = return_logs
        self.weights = {}

        # Register components
        for name, config in cfg_loss.items():
            self.loss_map[f"{prefix}{name}"] = instantiate(config["instance"])
            self.weights[f"{prefix}{name}"] = config["loss_weight"]

    def forward(self, net_out, gt):
        """Passes the FULL net_out and gt to every child loss.
        
        Args:
            net_out: Output from the network (BackboneOutput or GraphModelOutput).
            gt: Ground truth data (BatchData or FlatGraphData).
        """
        total_loss = 0.0
        logs = {}

        for name, loss_fn in self.loss_map.items():
            weight = self.weights[name]
            
            # The child loss is responsible for picking what it needs
            val = loss_fn(net_out, gt)

            total_loss += weight * val
            logs[name] = val.detach()

        output = (total_loss, logs) if self.return_logs else total_loss

        return output
    
    def clone(self, prefix: str = "") -> "CompositeLoss":
        """Make a copy of the class."""
        return self.__class__(deepcopy(self.cfg_loss), self.return_logs, prefix)