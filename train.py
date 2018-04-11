import argparse

import length.functions as F

from length.data_sets import MNIST, FashionMNIST
from length.models import MLP
from length.optimizers import Adam, SGD


def main(args):
    data_set = None
    if args.data_set == "mnist":
        data_set = MNIST(args.batch_size)
    if args.data_set == "fashion":
        data_set = FashionMNIST(args.batch_size)

    optimizer = None
    if args.optimizer == "adam":
        optimizer = Adam(args.learning_rate)
    if args.optimizer == "sgd":
        optimizer = SGD(args.learning_rate)

    model = MLP()

    for epoch in range(args.num_epochs):
        for iteration, batch in enumerate(data_set.train):
            model.forward(batch)
            model.backward(optimizer)

            if iteration % args.train_verbosity == 0:
                accuracy = F.accuracy(model.predictions, batch.labels).data
                print("train: epoch: {: 2d}, loss: {: 5.2f}, accuracy {:.2f}, iteration: {: 4d}".
                      format(epoch, model.loss.data, accuracy, iteration), end="\r")

        print("\nrunning test set...")
        sum_accuracy = 0.0
        sum_loss = 0.0
        for iterations, batch in enumerate(data_set.test):
            model.forward(batch, train=False)
            sum_accuracy += F.accuracy(model.predictions, batch.labels).data
            sum_loss += model.loss.data
        nr_batches = iterations - 1
        print(" test: epoch: {: 2d}, loss: {: 5.2f}, accuracy {:.2f}".
              format(epoch, sum_loss / nr_batches, sum_accuracy / nr_batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple neural network")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer to use")
    parser.add_argument("--data-set", choices=["mnist", "fashion"], default="mnist", help="Which data set to train for")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="(Initial) learning rate")
    parser.add_argument("--train-verbosity", type=int, default=50,
                        help="Interval (iterations) of printing training accuracy")

    main(parser.parse_args())
