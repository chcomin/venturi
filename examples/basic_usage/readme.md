This example shows a basic usage of venturi for training an image segmentation model. Typically, an experiment is divided into two parts: the definition of Python **objects** (dataset, model, metrics, etc) and the **parameters** required for instantiating the objects. As such, this example contains:

- scripts folder: Contains the definition of the dataset, the model and the performance metrics used for segmentation.
- config folder: Contains the base configuration file from venturi and the custom configuration parameters of the dataset, model, and metrics.
- experiment.py: Entry point of the experiment.

The torchvision package is required to run the scripts.