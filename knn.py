from enum import Enum
from math import floor
from typing import Optional

import numpy as np
from arg_parser import Method, setup_arg_parser
from utils import (
    ImageData,
    evaluate_data,
    get_data_copy,
    get_structured_data,
    twod_min_select,
    upsert_key,
    validate_model,
)


class WeightType(Enum):
    CONSTANT = 0
    LINEAR = 1
    RECIPROCAL = 2


class DistType(Enum):
    MANHATTAN = 0
    EUCLEDIAN = 1
    THIRD_POWER = 2


class DistObject:
    def __init__(self, state: str, dist: float):
        self.state = state
        self.dist = dist


class KNNParamCombo:
    def __init__(
        self,
        k: int,
        dist_type: DistType,
        weight_type: WeightType,
        len_div: int,
        conv_size: int,
    ):
        self.k = k
        self.dist_type = dist_type
        self.weight_type = weight_type
        self.len_div = len_div
        self.conv_size = conv_size

    def __str__(self) -> str:
        return (
            f"K: {self.k}, wei: {self.weight_type.value}, dis: {self.dist_type.value},"
            + f" lin: {self.len_div}, cnv: {self.conv_size}"
        )


# Default value, will get overwritten:
source_params: KNNParamCombo = KNNParamCombo(0, 0, 0, 0, 0)
source_data: list[ImageData] = []


def process_data(imgs: list[ImageData], p: KNNParamCombo):
    if p.conv_size > 0:  # 2D dimension reduction
        for i in range(len(imgs)):
            imgs[i].data = twod_min_select(imgs[i].data, p.conv_size)
    elif p.len_div > 0:  # 1D dimension reduction
        for i in range(len(imgs)):
            new_arr = []
            for j in range(0, len(imgs[i].data) - p.len_div, p.len_div):
                new_arr.append(min(imgs[i].data[j : j + p.len_div]))
            imgs[i].data = np.array(new_arr)


def process_all_data(
    train_images: list[ImageData],
    val_images: list[ImageData],
    test_images: Optional[list[ImageData]],
    p: KNNParamCombo,
):
    process_data(train_images, p)
    process_data(val_images, p)
    if test_images is not None:
        process_data(test_images, p)


def train(train_images_cp, params):
    global source_data, source_params
    source_data = train_images_cp
    source_params = params


def eval_image(val_image: ImageData):

    def get_image_distance(im1: ImageData, im2: ImageData):
        diff = im1.data - im2.data
        if source_params.dist_type == DistType.EUCLEDIAN:
            return np.sqrt(np.sum(np.square(diff)))
        elif source_params.dist_type == DistType.MANHATTAN:
            return np.sum(np.abs(diff))
        elif source_params.dist_type == DistType.THIRD_POWER:
            return np.cbrt(np.sum(np.power(np.abs(diff), 3)))

    closest: list[DistObject] = []
    for t in source_data:
        dist = get_image_distance(val_image, t)
        # This "remembers" the k closest values:
        if len(closest) < source_params.k or dist < closest[-1].dist:
            closest.insert(0, DistObject(t.state, dist))
            closest.sort(key=lambda a: a.dist)
            if len(closest) > source_params.k:
                closest.pop()

    # Get weight of each state from K nearest neighbors:
    state_counts = {}
    for cl in closest:
        upsert_key(cl.state, state_counts, 0)
        state_counts[cl.state] += (
            1
            if source_params.weight_type == WeightType.CONSTANT
            else (
                -cl.dist
                if source_params.weight_type == WeightType.LINEAR
                else (
                    1 / (cl.dist + 1)
                    if source_params.weight_type == WeightType.RECIPROCAL
                    else 0
                )
            )
        )

    # a[1] is probability, max(...)[0] is the state
    return max(state_counts.items(), key=lambda a: a[1])[0]


def tune_hyperparams(
    param_combos: list[KNNParamCombo], truth_dict, train_images, val_images, quick: bool
):
    max_acc, max_params = 0, None

    for p in param_combos:
        train_images_cp, val_images_cp, _ = get_data_copy(train_images, val_images)

        process_all_data(train_images_cp, val_images_cp, None, p)

        # If online, use less data to speed process up
        train_data = (
            train_images_cp[: floor(len(train_images_cp) / 2)]
            if quick
            else train_images_cp
        )
        train(train_data, p)

        acc = validate_model(truth_dict, val_images_cp, eval_image)
        max_params = p if acc > max_acc else max_params
        max_acc = max(acc, max_acc)
        print(
            f"[{str(p)}] -> Acc: {acc:.3f};; Max: [{str(max_params)}] -> Acc: {max_acc:.3f}"
        )

    return max_params


def tune_hyperparams_quick(truth_dict, train_images, val_images, k_override):
    """
    Online hyperparameter tuning to be used when submitted. Runs just a few
    combinations (on a smaller amount of training data?) (which were
    pre-selected when offline testing) to select the best one for the current data.
    """
    param_combos: list[KNNParamCombo] = []
    for k_val in range(3, 6, 1):  # K in K-NN
        for dist_type in range(2, 3, 1):  # Distance type
            for weight_type in range(2, 3, 2):  # Weight type
                # for k in range(2, 16, 2):  # 1D downscaling
                #     param_combos.append(KNNParamCombo(i, l, j, k, 0))
                for len_div in range(2, 3, 1):  # 2D downscaling
                    param_combos.append(
                        KNNParamCombo(
                            k_override if k_override is not None else k_val,
                            DistType(dist_type),
                            WeightType(weight_type),
                            0,
                            len_div,
                        )
                    )

    return tune_hyperparams(param_combos, truth_dict, train_images, val_images, True)


def tune_hyperparams_full(truth_dict, train_images, val_images, k_override):
    """
    Offline hyperparameter tuning used when determining best model
    parameters before submitting.
    """
    param_combos: list[KNNParamCombo] = []
    for k_val in range(1, 8, 1):  # K in K-NN
        for dist_type in range(0, 3, 1):  # Distance type
            for weight_type in range(0, 3, 1):  # Weight type
                for len_div in range(2, 16, 2):  # 1D downscaling
                    param_combos.append(
                        KNNParamCombo(
                            k_override if k_override is not None else k_val,
                            DistType(dist_type),
                            WeightType(weight_type),
                            len_div,
                            0,
                        )
                    )
                for len_div in range(1, 4, 1):  # 2D downscaling
                    param_combos.append(
                        KNNParamCombo(
                            k_override if k_override is not None else k_val,
                            DistType(dist_type),
                            WeightType(weight_type),
                            0,
                            len_div,
                        )
                    )

    print(f"Tuning with {len(param_combos)} different param combinations.")
    return tune_hyperparams(param_combos, truth_dict, train_images, val_images, False)


def train_test(
    train_images: list[ImageData],
    val_images: list[ImageData],
    test_images: list[ImageData],
    out_file: str,
    p: Optional[KNNParamCombo],  # Best parameters
):
    """Train the model and evaluate test data."""
    # Get fresh copy of data to process and reset likelihoods table
    train_images_cp, val_images_cp, test_images_cp = get_data_copy(
        train_images, val_images, test_images
    )

    process_all_data(train_images_cp, val_images_cp, test_images_cp, p)
    # Train on train+validation data since hyperparameters are fixed at this point.
    # truth_dict contains values for validation data as well.
    train(train_images_cp + val_images_cp, p)

    # Evaluate (this also saves output file)
    evaluate_data(test_images_cp, out_file, eval_image)


def main():
    parser = setup_arg_parser(method=Method.KNN)
    args = parser.parse_args()

    print(
        f"Runnning {args.k if args.k is not None else 'K'}-NN -> Directories:"
        + f" [Train: {args.train_path}, test: {args.test_path}], Output: {args.o}"
    )

    # Load all required data (shared across both classifiers)
    truth_dict, train_images, val_images, test_images = get_structured_data(args)

    # modes 2 and 3 are for testing (full hyperparam scanning and specific set)
    mode = 1
    if mode == 1:
        # Quickly tune hyperparameters based on data (e.g. 10x10 vs 28x28)...
        best_params = tune_hyperparams_quick(
            truth_dict, train_images, val_images, args.k
        )
        # Do final training and evaluate test data
        train_test(train_images, val_images, test_images, args.o, best_params)
    elif mode == 2:
        # Tune hyperparameters
        best_params = tune_hyperparams_full(
            truth_dict, train_images, val_images, args.k
        )
        # Do final training and evaluate test data
        train_test(train_images, val_images, test_images, args.o, best_params)
    else:
        # Tune hyperparameters
        best_params = KNNParamCombo(1, 7, 1)
        if args.k is not None:
            best_params.k = args.k
        # Do final training and evaluate test data
        train_test(train_images, val_images, test_images, args.o, best_params)


if __name__ == "__main__":
    main()
