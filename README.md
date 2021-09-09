# Poor Man's Transformers

Advanced Deep Learning from the ground-up.

The idea of this repository is to implement all the necessary layers of a transformer using only `numpy` for learning
purposes. The final goal is to train a Transformer model on [QQP](https://www.kaggle.com/c/quora-question-pairs) or a
model that performs NER decently. I was inspired by
[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch),
the [Advanced Machine Learning Specialization](https://www.coursera.org/specializations/aml), and
the [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing).

## Development logs

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
handle information flow (serial, parallel, concatenations). At the same time, I wouldn't like to waste time building a
flexible and feature-rich framework since we have PyTorch, TensorFlow Keras,
and [Google Trax](https://github.com/google/trax) for that.

To keep this toy "framework" as simple as possible, I want to minimize the number of base-classes: `Layers`,
`TrainableParameter` (used in `Layers` with an associated `Optimizer`), `Model`, and `Loss`. Activation functions will
be implemented as `Layer` objects. This simplification comes with its costs, of course, in terms of memory usage: more
"intermediate" tensors will be stored. When using `Dense` and `ReLU`, for example, both the linear combination vector
and the rectified vectors will be stored in memory.

Here's the list of objects I implemented:

#### Layer

A layer performs the forward propagation, for which it receives an input `x` and uses its `TrainableParameter`s to
compute the output. It also performs the backward propagation for it which receives the accumulated gradient `grad`
which represents `d_loss / d_layer` to propagate `d_loss / d_x = d_loss / d_layer * d_layer / d_x` and also the
input `x` used in the forward step to compute the gradients with respect to the parameters
`d_loss / d_param = d_loss / d_layer * d_layer / d_param`. It calls an update method for all `TrainableParameters` which
use an `Optimizer` to update their weights. Finally, the accumulated gradient is passed to continue the backward
propagation.

#### TrainableParameter

#### Optimizer

#### Loss

#### Model

### Embeddings

My next goal is to have an `Embedding` layer implemented. Even try to train a Continuous Bag of Words (CBOW) model and
try to replicate [word2vec]().