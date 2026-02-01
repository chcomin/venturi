"""Entry point of the experiment."""

from venturi import Config, Experiment

if __name__ == "__main__":
    vcfg = Config("config/base_config.yaml")
    vcfg.update_from_yaml("config/custom_config.yaml")

    experiment = Experiment(vcfg)

    final_metric = experiment.fit()
