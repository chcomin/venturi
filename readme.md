<div align="center">

<img src="assets/logo.png" alt="Venturi Logo" width="160">

# Venturi

**A hackable blueprint for training neural networks.**

[![PyPI](https://img.shields.io/pypi/v/venturi?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/venturi/)
[![Python](https://img.shields.io/pypi/pyversions/venturi?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/venturi/)
[![License](https://img.shields.io/pypi/l/venturi?style=flat-square)](https://github.com/chcomin/venturi/blob/main/LICENSE)

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#examples">Examples</a>
</p>

</div>

---

**A hackable blueprint for training neural networks using PyTorch and Lightning.**

Venturi is a minimalist alternative to Hydra and LightningCLI. It prioritizes code transparency and flexibility. By separating experiment parameters from logic, it enables complex experiment registration without the bloat of heavy configuration frameworks.

## Why Venturi?

Most configuration frameworks force you to learn their specific DSL or hide logic behind complex abstractions. Venturi takes a different approach:

* **Auditable Core:** The entire package logic resides in just two files: `config.py` and `core.py`. You can read, understand, and modify the inner workings.
* **Zero-Overhead Configuration:** No enforced `argparse` or `pydantic` validation by default. You instantiate Python objects directly from YAML. Validation is opt-in, not mandatory.
* **Global Context:** The full YAML configuration is passed to the main classes used for training. This allows you to define complex relationships (e.g., dynamically setting model depth based on dataset size) without changing the training loop.
* **Inheritance-First Design:** The experiment lifecycle is defined by classes designed to be subclassed when custom training logic is necessary.

## Installation

Install the core package:

```bash
pip install venturi
```

To run the provided examples, install the optional dependencies:

```bash
pip install "venturi[examples]"
```

## Quick Start

### 1. Scaffold a Project
Generate a standard directory structure and a default configuration file:

```bash
venturi create path/to/project
```

This creates a [base_config.yaml](venturi/base_config.yaml) file containing the default blueprints.

### 2. Define Your Experiment
Venturi uses a hierarchical configuration system. You load a base configuration and override it with experiment-specific YAMLs.

```python
from venturi import Config, Experiment

# 1. Load the project defaults
args = Config("base_config.yaml")

# 2. Overlay custom experiment parameters
args.update_from_yaml("experiments/my_custom_config.yaml")

# 3. Initialize and Run
experiment = Experiment(args)
experiment.fit()
```

## Examples

Some examples are provided in the examples directory:

| Example | Description |
| :--- | :--- |
| **[Configuration](examples/0_start_here)** | The core concept: instantiating arbitrary Python objects directly from YAML and mixing multiple config files |
| **[Basic Usage](examples/basic_usage)** | A complete image segmentation experiment setup |
| **[Base Config](venturi/base_config.yaml)** | The reference file describing all standard Venturi parameters |

## Design Philosophy

Venturi is built on the principle that **research code should be hackable**.

1.  **Transparency:** You should not have to dig through a call stack of 50 internal functions to understand how your configurations are parsed.
2.  **Flexibility:** If you want to add Pydantic validation, you can add it *before* passing the config to the `Experiment` class. It is not baked into the core.
3.  **Portability:** By avoiding complex CLI dependency injection, your experiments remain standard Python scripts that are easy to debug and deploy.