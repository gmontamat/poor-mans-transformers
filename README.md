# Poor Man's Transformers

Advanced Deep Learning from the ground-up.

The idea of this repository is to implement all the necessary layers of a transformer using just `numpy` for learning
purposes. The end goal is to train a Transformer model on [QQP](https://www.kaggle.com/c/quora-question-pairs) or a
model that performs Named Entity Recognition (NER) decently. I was inspired by
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
LogSoftmax), custom loss functions (sum binary cross-entropy and categorical cross-entropy as used for training BERT)
and handle input & output flow (serial, parallel, concatenations). At the same time, I wouldn't like to waste time
building a flexible and feature-rich framework since we already have PyTorch, TensorFlow with Keras,
[JAX](https://github.com/google/jax), and [Google Trax](https://github.com/google/trax) for that.

To keep this toy "framework" as simple as possible, I want to minimize the number of base classes. I ended up with:
`Layer`, `Parameter` (used in a `Layer` with an associated `Optimizer`), `Trainer`, and `Loss`. `Activation` functions
are a subclass of a `Layer` object. This simplification comes with its costs, of course, in terms of RAM usage: more
"intermediate" tensors will be stored in memory. When training a `Dense` layer with a `ReLU` activation, for example,
both the linear combination and the rectified (`max(X, 0)`) tensors will be stored in memory. I do not intend to run
this framework on a GPU, so RAM usage is not a big concern right now. Each layer will implement its backpropagation
step, the derivatives with respect to each parameter (Jacobian matrix) have to be computed because I don't want to
implement a tool such as [Autograd](https://github.com/hips/autograd) to do this automatically.

Here's the list of objects I implemented:

#### :pushpin: Layer and Activation

A layer performs two operations: forward propagation and backward propagation. For doing the forward pass, it receives
an input batch `X` and uses its `Parameter`s to compute the output batch. And in the case of the backward pass, it
receives the accumulated gradient `grad` (which represents the derivatives `d_loss / d_layer` for each element in the
batch) to compute and propagate to the previous layer: `d_loss / d_input = d_loss / d_layer · d_layer / d_input`. It
also receives the input batch `X` used in the forward step to compute the gradients with respect to the
parameters `d_loss / d_parameter = d_loss / d_layer · d_layer / d_parameter`. Next, it calls the `update` method on
all `Parameter`s which use an `Optimizer` instance to update their weights. Finally, the accumulated
gradient `d_loss / d_input` is returned to proceed with the network's backward propagation.

When instantiated, an `input_shape` and `output_shape` could be set, or else they will be set by the `Trainer` during
the model's initialization step. The initial weights of each `Parameter` also need to be defined during this step.

An `Activation` is a special type of `Layer` whose `input_shape` and `output_shape` are the same.

:heavy_check_mark: [Layer](poormanstransformers/layers.py#L32-L77)
:heavy_check_mark: [Activation](poormanstransformers/layers.py#L80-L90)
:white_check_mark: [Dense](poormanstransformers/layers.py#L93-L131)
:white_check_mark: [ReLU](poormanstransformers/layers.py#L134-L140)
:white_check_mark: [Softmax](poormanstransformers/layers.py#L143-L161)
:white_check_mark: [LogSoftmax](poormanstransformers/layers.py#L164-L181)
:white_check_mark: [Dropout](poormanstransformers/layers.py#L184-L202)

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

:heavy_check_mark: [Loss](poormanstransformers/losses.py#L6-L27)
:heavy_check_mark: [Metric](poormanstransformers/losses.py#L48-L59)
:white_check_mark: [CategoricalCrossEntropy](poormanstransformers/losses.py#L30-L45)
:white_check_mark: [Accuracy](poormanstransformers/losses.py#L62-L70)

#### :pushpin: Model, Trainer, and DataGeneratorWrapper

Instead of following Keras-style Sequential Model and the `model.compile()` method to define the optimizer, loss, and
metrics, a `Model` in this framework is just a list of `Layer` instances (I think this will help us handle complex flows
with stack operations). Hence, I defined the `Trainer` which receives a model, optimizer, loss, learning rate schedule,
early stopping, and metrics to run the supervised training with training and evaluation data generators. This approach
resembles the Trax framework more than Keras or PyTorch.

The `fit` method in `Trainer` is the key function of this class. It prepares the model by setting and validating the
`input_shape` and `output_shape` for every layer, and initializing the layer's weights. Training and evaluation data is
passed via a generator function that has to be written for every particular dataset and needs to be wrapped using the
`DataGeneratorWrapper` whose only purpose is to initialize the generator with all the arguments passed so that the data
could be "rewound" at the beginning of each epoch.

:heavy_check_mark: [Trainer](poormanstransformers/train.py#L34)
:heavy_check_mark: [DataGeneratorWrapper](poormanstransformers/train.py#L16-L31)

#### :warning: Challenges

The most difficult part of this first step was to do the backwards propagation. I needed to compute Jacobian matrices of
several vector functions. The following articles helped me clarify the math needed:

* [Jacobian, Chain rule and backpropagation](https://suzyahyah.github.io/calculus/machine%20learning/2018/04/04/Jacobian-and-Backpropagation.html)
* [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

#### :gem: Sample code

:heavy_check_mark: [MLP for MNIST Digit recognition](./examples/mlp.py)

```shell
python ./examples/mlp.py
```

### :bookmark: Word Embeddings

My next goal is to have an `Embedding` layer implemented and try it out with a Continuous Bag of Words (CBOW) model. We
can replicate embeddings like those in [word2vec](https://arxiv.org/pdf/1301.3781.pdf). With the framework in place and
validated with the Multilayer Perceptron trained on MNIST, this part should be a matter of adding the necessary
subclasses and helper functions.

#### :pushpin: Embedding

The `Embedding` layer is equivalent to a `Dense` layer if we converted the word representations (numbers in the range
`[0, vocab_size)`) to their one-hot representation and performed a matrix-matrix dot product between the input and
weights. Here instead, the layer takes the word representation (integer between 0 and `vocab_size-1`) and use it to
index the weights' matrix. We avoid doing a matrix-matrix dot product which is more expensive.

:white_check_mark: [Embedding](poormanstransformers/layers.py#L205-L237)

#### :pushpin: AxisMean

The CBOW model works by averaging the embeddings of a context window surrounding the target word. The dimension average
is usually done by a `Lambda` layer which takes a lambda function and use it as the forward propagation step. Frameworks
have tools like [autograd](https://github.com/HIPS/autograd) to compute a gradient (formally, *jacobian*) given the
forward function. For simplicity, I created the `AxisMean` layer instead of a `Lambda` layer which doesn't require the
aforementioned tool.

:white_check_mark: [AxisMean](poormanstransformers/layers.py#L240-L260)

#### :pushpin: RMSProp

Implementing this optimizer is straightforward. Just need to keep a moving average of the element-wise squared gradient
and use its squared root when updating the weights.

:white_check_mark: [RMSProp](poormanstransformers/optimizers.py#L46-L65)

#### :warning: Challenges

Implementing backpropagation for the `Embedding` layer was a bit tricky but not as hard as the Softmax and LogSoftmax
layers. The following resources guided me through this step:

* [What is the difference between an Embedding Layer and a Dense Layer?](https://stackoverflow.com/questions/47868265/what-is-the-difference-between-an-embedding-layer-and-a-dense-layer)
* [Implementing Deep Learning Methods and Feature Engineering for Text Data: The Continuous Bag of Words (CBOW)](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)
* [Back propagation in an embedding layer](https://medium.com/@ilyarudyak/back-propagation-in-an-embedding-layer-30382fa7f023)

#### :gem: Sample code

:heavy_check_mark: [Continuous Bag of Words (CBOW) with Wikipedia Sentences](./examples/cbow.py)

```shell
python ./examples/cbow.py
```
