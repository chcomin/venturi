from torch import nn

from torchtrainer.engine.config import Config, instantiate


class CompositeLoss(nn.Module):
    """Container class to combine multiple loss functions."""
    def __init__(self, cfg_loss: Config, return_logs: bool = True):
        """Args:
        loss_components: Dict structure from the builder:
        {
            'name_for_logging': {'loss_fn': nn.Module, 'weight': float},
            ...
        }.
        """
        super().__init__()
        self.loss_map = nn.ModuleDict()
        self.return_logs = return_logs
        self.weights = {}

        # Register components
        for name, config in cfg_loss.items():
            self.loss_map[name] = instantiate(config["instance"])
            self.weights[name] = config["weight"]

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
            logs[f"loss_{name}"] = val.detach()

        output = (total_loss, logs) if self.return_logs else total_loss

        return output
