# Poor Man's Transformers

Advanced Deep Learning from the ground-up.

The idea of this repository is to implement all the necessary layers of a transformer using only `numpy` for learning
purposes. The final goal is to train a Transformer model on [QQP](https://www.kaggle.com/c/quora-question-pairs) or a
model that performs NER decently. I was inspired by
[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch),
the [Advanced Machine Learning Specialization](https://www.coursera.org/specializations/aml), and
the [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing).

## :notebook: Development log

I'm keeping track of my progress in this section, so it can be used for future reference when learning Deep Learning
from the very beginnings.

### :bookmark: First steps: basic layers and training framework

First things first, I need to implement the basic structure of the framework and be able to train a Multilayer
Perceptron (MLP) with it. I base my work
on [ML-From-Scratch Deep Learning implementation](https://github.com/eriklindernoren/ML-From-Scratch#deep-learning)
and an assignment from the [Introduction to Deep Learning](https://www.coursera.org/learn/intro-to-deep-learning)
course.

Even though this was supposed to be an easy step, I ended up spending a lot of time on it trying to come up with the
simplest OOP architecture possible. New layers have to be easy to code, and I also want to experiment with different
optimizers (SGD, Adam, RMSProp), learning rate schedules, activation functions (ReLU, tanh, sigmoid, Softmax,
LogSoftmax), custom loss functions (sum binary cross-entropy and categorical cross-entropy as they do in BERT) and
handle input & output flow (serial, parallel, concatenations). At the same time, I wouldn't like to waste time building
a flexible and feature-rich framework since we have PyTorch, TensorFlow with Keras,
and [Google Trax](https://github.com/google/trax) for that.

To keep this toy "framework" as simple as possible, I want to minimize the number of base-classes. I ended up with:
`Layer`, `Parameter` (used in a `Layer` with an associated `Optimizer`), `Trainer`, and `Loss`. `Activation` functions
are a subclass of a `Layer` object. This simplification comes with its costs, of course, in terms of memory usage: more
"intermediate" tensors will be stored. When using `Dense` and `ReLU`, for example, both the linear combination vector
and the rectified (`max(X, 0)`) vectors will be stored in memory. I do not intend to run it on a GPU, so RAM is not a
big concern right now.

Here's the list of objects I implemented:

#### :pushpin: Layer and Activation

A layer performs two operations: forward propagation and backward propagation. For doing the forward pass, it receives
an input `X` and uses its `Parameter`s to compute the output. And in the case of the backward pass, it receives the
accumulated gradient `grad` (which represents `d_loss / d_layer`) to compute and propagate to the previous layer
`d_loss / d_input = d_loss / d_layer * d_layer / d_input`. It also receives the input `X` used in the forward step to
compute the gradients with respect to the parameters `d_loss / d_parameter = d_loss / d_layer * d_layer / d_parameter`.
Next, it calls the `update` method on all `Parameter`s which use an `Optimizer` instance to update their weights.
Finally, the accumulated gradient `d_loss / d_input` is returned to proceed with the network's backward propagation.

When instantiated, an `input_shape` and `output_shape` can be defined, or they will be set by the `Trainer` during the
model initialization step. The initial weights of each `Parameter` also need to be defined during this step.

An `Activation` is a special type of `Layer` whose `input_shape` and `output_shape` are the same.

:heavy_check_mark: [Layer](poormanstransformers/layers.py#L32-L75)
:heavy_check_mark: [Activation](poormanstransformers/layers.py#L78-L87)
:white_check_mark: [Dense](poormanstransformers/layers.py#L90-L128)
:white_check_mark: [ReLU](poormanstransformers/layers.py#L131-L137)
:white_check_mark: [Softmax](poormanstransformers/layers.py#L140-L150)
:white_check_mark: [LogSoftmax](poormanstransformers/layers.py#153-L161)
:white_check_mark: [Dropout](poormanstransformers/layers.py#L164-L183)

#### :pushpin: Parameter and Optimizer

A `Parameter` is instantiated only by a `Layer`. Its weights can be accessed by calling the `Parameter` instance and are
updated during back-propagation using the `update` method. For `update` to be called, an `Optimizer` instance needs to
be set and its initial weights need to be defined during the `Layer`'s initialization.

Each `Parameter` instantiated in the framework will have a copy of an `Optimizer` instance with the properties defined
by the `Trainer` object. The `Optimizer` is in charge of updating the parameter's value and may store auxiliary
variables to do so, hence, each parameter has a unique copy of it. Again, it set by the `Trainer` during the model's
initialization.

:heavy_check_mark: [Parameter](poormanstransformers/layers.py#L8-L29)
:heavy_check_mark: [Optimizer](poormanstransformers/optimizers.py#L4-L11)
:white_check_mark: [Adam](poormanstransformers/optimizers.py#L14-L43)

#### :pushpin: Loss and Metric

These classes are pretty straightforward: instances are called with the ground truth `y` and predictions `y_hat` (or
prediction's probabilities *logits*) and return the calculated metric. The `Loss` class also returns the gradient
`d_loss / d_yhat` to begin the backward propagation.



#### :pushpin: Model, Trainer, and DataGeneratorWrapper

Instead of following Keras-style Sequential Model and the `model.compile()` method to define the optimizer, loss, and
metrics, a `Model` in this framework is just a list of `Layer` instances (I think this will help us handle complex flows
with stack operations). Hence, I defined the `Trainer` which receives a model, optimizer, loss, learning rate schedule,
early stopping, and metrics to run the supervised training with training and evaluation data generators. This approach
resembles the Trax framework more than Keras or PyTorch.

The `fit` method in the `Trainer` is the key one in this class. It prepares the model by setting and validating the
`input_shape` and `output_shape` for every layer, and initializing the layer's weights. Next, given a.

#### :construction: Example code

:heavy_check_mark: MLP with MNIST Digit recognition

### :construction: Embedding layer

My next goal is to have an `Embedding` layer implemented. Even try to train a Continuous Bag of Words (CBOW) model and
replicate [word2vec](https://arxiv.org/pdf/1301.3781.pdf).
