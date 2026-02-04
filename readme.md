<div align="center">

<img src="https://raw.githubusercontent.com/chcomin/venturi/master/assets/logo.png" alt="Venturi Logo" width="160">

# Venturi

**A hackable blueprint for training neural networks**

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#installation">Installation</a> •
  <a href="#why-venturi">Why Venturi?</a> •
  <a href="#examples">Examples</a>
</p>

</div>

---


Venturi is a minimalist alternative to Hydra and LightningCLI for separating experiment parameters from code, while prioritizing framework transparency and flexibility. 

## Quick Start

### Scaffold a Project
Generate a default configuration file:

```bash
venturi create path/to/project
```

This creates a [base_config.yaml](venturi/base_config.yaml) file containing the default parameters for an experiment.

### Add custom configuration and run an experiment

```python
from venturi import Config, Experiment

# Load Venturi defaults
vcfg = Config("base_config.yaml")

# Add custom configuration for dataset, model and loss function
vcfg.update_from_yaml("experiments/my_custom_config.yaml")

# Initialize, train and test. All training artifacts, including
# performance metrics and model checkpoints are logged.
experiment = Experiment(vcfg)
experiment.fit()
results = experiment.test()
```

### Optimize experiment hyperparameters

```python

vcfg = ... # Load configuration
experiment = Experiment(vcfg)

# Load hyperparameter search space
vcfg_space = Config("config/search_space.yaml")
# Run hyperparameter search
study = experiment.optimize(vcfg_space)

print("Best parameters:", study.best_params)
```

### Change a core Venturi component

The default experiment lifecycle is flexible to support changing the main training artifacts (dataset, model, metrics and composite loss functions). But if custom training logic is required, you can just change one of the base classes:

```python
from venturi import TrainingModule, Experiment

class CustomTrainingModule(TrainingModule):
  def training_step(self, batch):
    # Get all experiment settings
    vcfg = self.vcfg
    # Create custom logic for training step
    ...

class CustomExperiment(Experiment):
  def get_loggers(self):
    vcfg = self.vcfg
    # Add your own logic for storing performance metrics
    ...

...
experiment = CustomExperiment(vcfg)
experiment.fit()
```

## Installation

Install the core package:

```bash
pip install venturi
```

To enable Weights & Biases logging and to run the provided examples, install all dependencies:

```bash
pip install "venturi[all]"
```

There is no conda package yet. To install on conda you can do

```bash
conda env create -n env_name -f environment.yaml 
conda activate env_name
# --no-build-isolation --no-deps is useful for avoiding pip conflicts
pip install --no-build-isolation --no-deps venturi 
```

## Why Venturi?

Most configuration frameworks force you to learn their specific DSL or hide logic behind complex abstractions. Venturi takes a different approach:

* **Auditable Core:** The entire package logic resides in just two files: `config.py` and `core.py`. You can read, understand, and modify the inner workings.
* **Zero-Overhead Configuration:** No enforced `argparse` or `pydantic` validation by default. You instantiate Python objects directly from YAML. Validation is opt-in, not mandatory.
* **Global Context:** The full YAML configuration is passed to the main classes used for training. This allows you to define complex relationships (e.g., dynamically setting model depth based on dataset size) without changing experiment setup code.
* **Inheritance-First Design:** The experiment lifecycle is defined by classes designed to be subclassed when custom training logic is necessary.


## Examples

Some examples are provided in the examples directory:

| Example | Description |
| :--- | :--- |
| **[Configuration](examples/0_start_here)** | The core concept: instantiating arbitrary Python objects directly from YAML and mixing multiple config files |
| **[Image segmentation](examples/image_segmentation)** | A complete image segmentation experiment setup |
| **[Hyperparameter search](examples/hyperparameter_search)** | Hyperparameter optimization using Optuna |
| **[Base Config](venturi/base_config.yaml)** | The reference file describing all standard Venturi parameters |

## Design Philosophy

Venturi is built on the principle that **research code should be hackable**.

1.  **Transparency:** You should not have to dig through a call stack of 50 internal functions to understand how your configurations are parsed.
2.  **Flexibility:** If you want to add Pydantic validation, you can add it *before* passing the config to the `Experiment` class. It is not baked into the core.
3.  **Portability:** By avoiding complex CLI dependency injection, your experiments remain standard Python scripts that are easy to debug and deploy.