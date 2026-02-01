"""Entry point of the experiment."""

from venturi import Config, Experiment

if __name__ == "__main__":
    vcfg = Config("config/base_config.yaml")
    vcfg.update_from_yaml("config/custom_config.yaml")

    experiment = Experiment(vcfg)

    # Train the model
    final_metric = experiment.fit()

    # Test the trained model. The test metrics are logged to csv and/or wandb if enabled. They
    # are also returned as a dictionary and printed on the terminal.
    results = experiment.test()

    
