import argparse
from enum import Enum


class Method(Enum):
    BAYES = 0
    KNN = 1


def setup_arg_parser(method: Method) -> argparse.ArgumentParser:
    description = (
        "Learn and classify image data with a naive bayes classifier."
        if method is Method.BAYES
        else "Learn and classify image data with a k-NN classifier."
    )
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "train_path", type=str, help="path to the training data directory"
    )
    parser.add_argument(
        "test_path", type=str, help="path to the testing data directory"
    )
    if method is Method.KNN:
        parser.add_argument(
            "-k",
            type=int,
            help="run k-NN classifier (if k is 0 the code selects best K",
        )

    parser.add_argument(
        "-o",
        metavar="filepath",
        default="classification.dsv",
        help="path (including the filename) of output .dsv file with results",
    )
    return parser
