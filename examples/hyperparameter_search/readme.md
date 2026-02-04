This example shows how to do hyperparameter search. The searched hyperparameters are:

* model: CNN or Vit
* For the CNN model:
  * Number of convolution layers: 2 to 10
  * Number of filters in the hidden layers: 16 to 64
* For the Vit model:
  * Number of encoder blocks
  * Emdedding size: 16 to 64
* optimizer: SGD or Adam
  * Learning rate: 1e-4 to 1e-1

The file [search_space.yaml](config/search_space.yaml) defines the search space above.

To simplify the example and show that the hyperparameter search works, a MockTrainingModule class is used in the experiment. The class changes the validation step to return a hand-crafted score that gets lower for specific hyperparameter combinations. In a real situation this would not be necessary and a real metric such as validation loss or accuracy would be used.