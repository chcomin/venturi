"""Use example configuration files to run some tests."""

from pathlib import Path
EXAMPLE_DIR = Path("../../../examples/basic_usage/")
import sys
sys.path.insert(0, str(EXAMPLE_DIR))

from venturi import Config, Experiment

if __name__ == "__main__":  

    vcfg = Config(EXAMPLE_DIR / "config" / "base_config.yaml")
    vcfg.update_from_yaml(EXAMPLE_DIR / "config" / "custom_config.yaml")
    vcfg.update_from_yaml("config/test_config.yaml")

    experiment = Experiment(vcfg)
    final_metric = experiment.fit()

    #experiment.test(checkpoint_name="best_model_epoch=7_val_loss=0.6083.ckpt")