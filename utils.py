from argparse import Namespace
import copy
from math import floor, sqrt
import os
from typing import Any, Callable, Optional
import numpy as np
from PIL import Image


class ImageData:
    """
    Class to hold image data as a numpy array, name of the original file and the
    state (either from truth table if known or the assigned one if part of test set.)
    """

    def __init__(
        self, name: str, data: np.ndarray[Any, np.dtype[Any]], state: Optional[str]
    ):
        self.name = name
        self.data = data
        self.state = state


def read_truth_file(file_path: str) -> dict[str, str]:
    """
    Converts the truth.dsv file at file_path to a dictionary, where the image
    file name is the key and the state is the value.
    """
    with open(file_path) as f:
        return {l.strip().split(":")[0]: l.strip().split(":")[1] for l in f.readlines()}


def get_truth_file_path(dir: str) -> str | None:
    """Gets the path of the first truth file in 'dir'."""
    dsv_files = list(filter(lambda a: ".dsv" in a, os.listdir(dir)))
    return os.path.join(dir, dsv_files[0]) if len(dsv_files) > 0 else None


def get_images(dir: str) -> tuple[str, str]:
    """Returns tuples of [image_file_name, path_to_image_incl_name]"""
    return list(
        map(
            lambda f: (f, os.path.join(dir, f)),
            filter(lambda a: ".png" in a, os.listdir(dir)),
        )
    )


def read_image_file(file_name: str, path: str, state: str):
    """
    Creates instance of ImageData class with file_name, image data and optionally state
    """
    return ImageData(file_name, np.array(Image.open(path)).astype(int).flatten(), state)


def get_structured_data(args: Namespace):
    """
    Reads taining and testing data from paths specified in args,
    splits training data into training and validation sets.
    """
    truth_path = get_truth_file_path(args.train_path)
    if truth_path is None:
        print("Error: No training truth file found in train_path.")
        exit(1)

    truth_dict = read_truth_file(truth_path)
    train_val_images = [
        read_image_file(i[0], i[1], truth_dict[i[0]])
        for i in get_images(args.train_path)
    ]

    train_images: list[ImageData] = []
    val_images: list[ImageData] = []

    for i in range(len(train_val_images)):
        if i % len(train_val_images) < 0.7 * len(train_val_images):
            train_images.append(train_val_images[i])
        else:
            val_images.append(train_val_images[i])

    test_images = [
        read_image_file(i[0], i[1], None) for i in get_images(args.test_path)
    ]
    return truth_dict, train_images, val_images, test_images


def twod_min_select(arr: np.ndarray[Any, np.dtype[Any]], kernel_size: int):
    """Reduces dimension of image in a '2D-aware way'."""
    # This is definitely not the prettiest way to do this, but it works...
    # The point is to select the minimum value from each
    # [kernel_size * kernel_size] area to downscale the image.
    # It's especially useful in KNN, where the reduced number
    # of dimensions improves results quite a lot.
    new_arr = []
    side_len = floor(sqrt(len(arr)))
    for y in range(0, side_len, kernel_size):
        for x in range(0, side_len, kernel_size):
            new_el = 255
            for yy in range(0, kernel_size):
                for xx in range(0, kernel_size):
                    i = (y + yy) * side_len + (x + xx)
                    if i < len(arr):
                        new_el = min(new_el, arr[i])

            new_arr.append(new_el)
    return np.array(new_arr)


def get_data_copy(train_images, val_images, test_images=None):
    """
    Gets deep copies of all data. Useful when testing multiple
    sets of hyperparameters to keep each test separated. test_images
    is optional, since it's not used when selecting hyperparameters.
    """
    return (
        copy.deepcopy(train_images),
        copy.deepcopy(val_images),
        None if test_images is None else copy.deepcopy(test_images),
    )


def validate_model(
    truth_dict: dict[str, str],
    val_images: list[ImageData],
    eval_fn: Callable[[ImageData], str],
):
    """
    Validates model using supplied truth dictionary, validation
    image dataset and evaluation function (which is the only thing that
    is unique for each model.)
    Keeps track of model success rate and returns it at the end.
    """
    right, wrong = 0, 0
    for im in val_images:
        match = eval_fn(im) == truth_dict[im.name]
        right += match
        wrong += not match

    return right / (right + wrong)


def evaluate_data(
    test_images: list[ImageData], out_file: str, eval_fn: Callable[[ImageData], str]
):
    """
    Classifies test data and saves output to the specified file. Uses eval_fn to
    get answer (which is the only thing that is unique for each model.)
    """
    for i in range(len(test_images)):
        test_images[i].state = eval_fn(test_images[i])

    with open(out_file, "w") as f:
        f.write("\n".join(list(map(lambda im: f"{im.name}:{im.state}", test_images))))
        f.write("\n")

def upsert_key(key: str | int, dict: dict[str | int, Any], val: Any):
    """
    Inserts key into dict if it's not already there and assigns val as the default value.
    If key is in dict, nothing happens.
    """
    if key not in dict:
        dict[key] = val
