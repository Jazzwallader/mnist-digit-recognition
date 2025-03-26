# Convolutional Neural Network for Hand-drawn Digit Recognition
![Digit Recognizer Demo](digit-recognizer-demo2.gif)

Above is a demo of this model. Digits 0 through 9 are drawn on the canvas by the user. Once the user executes the prediction, the canvas is processed into an image, scaled and inverted. The processed image is input into a CNN, which predicts the digit the user has drawn. 

The model was trained on the popular MNIST dataset, using a mixture of data augmentation, batch normalization, learning-rate warmup, and dropout regularization to achieve high accuracy when validated against both test data and new (user-generated) input.