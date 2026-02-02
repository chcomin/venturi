"""Use example configuration files to run some tests."""

import sys
from pathlib import Path

from venturi import Config, Experiment

EXAMPLE_DIR = Path("../../../examples/image_segmentation/")
sys.path.insert(0, str(EXAMPLE_DIR))

if __name__ == "__main__":  

    vcfg = Config(EXAMPLE_DIR / "config" / "base_config.yaml")
    vcfg.update_from_yaml(EXAMPLE_DIR / "config" / "custom_config.yaml")
    vcfg.update_from_yaml("config/test_config.yaml")

    experiment = Experiment(vcfg)
    #final_metric = experiment.fit()

    #experiment.test()
    experiment.test(checkpoint_name="last.ckpt")