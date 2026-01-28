# Venturi

A hackable blueprint for training neural networks using PyTorch and Lightning.

## Desigin principles

The configuration is purposely designed to not have pydantic validation. You create your classes and/or custom functions and add their parameters to a yaml file, and that is it. You can add your own pydantic validation before passing the configuration to the experiment object.