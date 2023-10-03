#!/bin/bash

if [ ! -e mnist.npz ]; then
  wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
fi
