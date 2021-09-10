# Poor Man's Transformers

Advanced Deep Learning from the ground-up.

The idea of this repository is to implement all the necessary layers of a transformer using only `numpy` for learning
purposes. The final goal is to train a Transformer model on [QQP](https://www.kaggle.com/c/quora-question-pairs) or a
model that performs NER decently. I was inspired by
[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch),
the [Advanced Machine Learning Specialization](https://www.coursera.org/specializations/aml), and
the [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing).

## Development log

I'm keeping track of my progress in this section, so it can be used for future reference when learning Deep Learning
from the very beginnings.

### First steps: basic layers and training framework

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

To keep this toy "framework" as simple as possible, I want to minimize the number of base-classes: `Layer`,
`Parameter` (used in `Layer`s with an associated `Optimizer`), `Trainer`, and `Loss`. Activation functions will be
implemented as `Layer` objects. This simplification comes with its costs, of course, in terms of memory usage: more
"intermediate" tensors will be stored. When using `Dense` and `ReLU`, for example, both the linear combination vector
and the rectified (`max(X, 0)`) vectors will be stored in memory.

Here's the list of objects I implemented:

#### Layer

A layer performs the forward propagation, for which it receives an input `X` and uses its `Parameters`s to compute the
output. It also performs the backward propagation for it which receives the accumulated gradient `grad`
which represents `d_loss / d_layer` to propagate `d_loss / d_X = d_loss / d_layer * d_layer / d_X` and also the
input `X` passed in the forward step to compute the gradients with respect to the parameters (`W`)
`d_loss / d_W = d_loss / d_layer * d_layer / d_W`. It calls an update method for all `Parameter`s which use
an `Optimizer` to update their weights. Finally, the accumulated gradient is returned to continue the backward
propagation process.

When instantiated, an `input_shape` can be defined, or it will be computed by the `Trainer` during the model
initialization step. The initial values for the `Parameter`s also need to be defined during this step.

#### Parameter

These objects are used only by `Layer` instances. Their value can be accessed by calling an instantiated object and are
updated during backpropagation using the `update()` method. For it to be called, an `Optimizer` needs to be instantiated
within this object and the initial values need to be defined in the `Layer`'s initialization.

#### Optimizer

Each `Parameter` instantiated in the framework will create an instance of this type with the properties defined in
the `Trainer`. It is in charge of updating the parameter's value and may store auxiliary variables to do so, hence, each
parameter has a unique variant of it.

#### Loss and Metric

These classes are pretty straightforward: instances are called with the ground truth `y` and predictions `y_hat` (or
prediction's probabilities *logits*) and return the calculated metric. The `Loss` class also returns the gradient
`d_loss / d_yhat` to begin the backward propagation.

#### Model and Trainer

Instead of following Keras-style Sequential Model and the `model.compile()` method to define the optimizer, loss, and
metrics, a `Model` in this framework is just a list of `Layer` instances (I think this will help us handle complex flows
with stack operations). Hence, I defined the `Trainer` which receives a model, optimizer, loss, learning rate schedule,
early stopping, and metrics to run the supervised training with training and evaluation data generators. This approach
resembles the Trax framework more than Keras or PyTorch.

### Embedding layer

My next goal is to have an `Embedding` layer implemented. Even try to train a Continuous Bag of Words (CBOW) model and
try to replicate [word2vec]().
