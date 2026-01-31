"""Entry point of the experiment."""

from venturi import Config, Experiment

if __name__ == "__main__":
    args = Config("config/base_config.yaml")
    args.update_from_yaml("config/custom_config.yaml")

    experiment = Experiment(args)

    final_metric = experiment.fit()
