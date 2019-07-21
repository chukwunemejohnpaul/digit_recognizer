## Recognizing handwritten digits with deep learning, python, opencv and keras


This model was trained on the MNIST database of handwritten digits, 
using a Lenet architecture but with a batchnormalization layer added to it.

The model was trained for 5 epochs peaking at an acurracy of 98.64% and a 
validation accuracy of 98.56%.

The performance of the model on the test images however, is far from perfect.
The reason this is so is because, the test images contain digits bent at different angles, 
and this was not acccounted for in the traing dataset.


The model can be improved by augmenting the image perphaps with keras's builtin 
ImageDataGenerator function, or with a custom augmenting function.
