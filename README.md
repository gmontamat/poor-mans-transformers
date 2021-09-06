# poor-mans-transformers

The idea of this repository is to implement all the necessary layers of a transformer using only `numpy` for learning
purposes. The final goal is to train a Transformer model on [QQP](https://www.kaggle.com/c/quora-question-pairs) or a
model that performs NER decently. I was inspired by
[ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch),
the [Advanced Machine Learning Specialization](https://www.coursera.org/specializations/aml), and
the [Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing).

## Development logs

I'm keeping track of my progress in this section

### First steps: basic layers and training framework

First things first, I need to implement the basic architecture of the framework and be able to train an MLP with it. I
base my work
on [ML-From-Scratch Deep Learning implementation](https://github.com/eriklindernoren/ML-From-Scratch#deep-learning)
and an assignment from the [Introduction to Deep Learning](https://www.coursera.org/learn/intro-to-deep-learning)
course.

Even though this was supposed to be an easy step, I ended up spending a lot of time on it trying to come up with the
simplest architecture possible. New layers have to be easy to code, and I also want to experiment with different
optimizers (SGD, Adam, RMSProp), learning rate schedules, activation functions (ReLU, tanh, sigmoid, Softmax,
LogSoftmax), custom loss functions (sum binary cross-entropy and categorical cross-entropy as they do in BERT) and
handle information flow (serial, parallel, concatenations). At the same time, I wouldn't like to waste time building a
flexible and feature-rich framework since we have PyTorch, TensorFlow Keras,
and [Google Trax](https://github.com/google/trax) for that.

### Embeddings

My first goal is to have an `Embedding` layer implemented. Even try to train a Continuous bag-of-words (CBOW) model and
try to replicate [word2vec]().