"""Example of hyperparameter search using Venturi."""
import torch

from venturi import Config, Experiment, TrainingModule


class MockTrainingModule(TrainingModule):
    """Mock training module that replaces the validation step to return a mock metric. The
    mock metric, defined in _get_score, is designed to have an optimum at specific hyperparameter 
    values.
    """

    def validation_step(self, batch):
        """Performs a validation step."""
        x, y = batch
        logits = self(x)
        loss, _ = self.val_loss(logits, y)

        # Get mock validation metric and log it
        val_metric_name = self.vcfg.training.validation_metric
        val_metric = self._get_score(self.trainer.current_epoch)
        self.log(val_metric_name, val_metric, on_step=False, on_epoch=True)

        if val_metric_name == "val/loss":
            loss = val_metric

        return loss
    
    def _get_score(self, epoch):
        """Mock objective.
         
        Pretend the optimal hyperparameters are 5 layers, 30 dim, 0.001 lr and the 
        performance improves over epochs but saturates at some point.
        """

        OPT_LAYERS = 5
        OPT_DIM = 30
        OPT_LR = 0.001

        vcfg_m = self.vcfg.model
        num_layers = vcfg_m.num_layers
        dim = vcfg_m.get("num_filters", 0) + vcfg_m.get("hidden_dim", 0)
        lr = self.vcfg.training.optimizer.lr

        # Add extra cost for the ViT architecture
        arch_cost = 10 if "vit" in vcfg_m.setup._target_ else 0

        val_metric = abs(num_layers - OPT_LAYERS) + abs(dim - OPT_DIM) + 1e2*abs(lr - OPT_LR)
        val_metric += arch_cost + 10/(min(epoch+1, 7))

        return torch.tensor(val_metric)
    
class MockExperiment(Experiment):
    """Mock experiment that uses the MockTrainingModule."""

    def get_model(self):
        """Override base TrainingModule."""
        return MockTrainingModule(self.vcfg)

if __name__ == "__main__":
    
    vcfg = Config("config/base_config.yaml")
    vcfg.update_from_yaml("config/exp_config.yaml")
    experiment = MockExperiment(vcfg)

    vcfg_space = Config("config/search_space.yaml")
    study = experiment.optimize(vcfg_space)

    print("Best parameters:", study.best_params)