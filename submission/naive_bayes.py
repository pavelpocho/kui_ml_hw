from typing import Any, Optional
import numpy as np
import time
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


class LHoodTableValue:
    def __init__(self, count, total):
        self.count = count
        self.total = total

    def inc_count(self):
        self.count += 1

    def inc_total(self):
        self.total += 1

    def get_prob(self):
        return self.count / self.total


class BayesParamCombo:
    def __init__(self, alpha, gray_vals, len_div, conv_size):
        self.alpha = alpha
        self.gray_vals = gray_vals
        self.len_div = len_div
        self.conv_size = conv_size

    def __str__(self) -> str:
        return f"K: {self.alpha:.3f}, GV: {self.gray_vals}, lnd: {self.len_div}, cnv: {self.conv_size}"


# cond_lhood_table[state][pixel_index][pixel_value] = probability
likehoods: dict[str, dict[int, dict[int, LHoodTableValue]]] = {}
max_value: int = 256
possible_states: list[str] = []
# Keep track of time so we don't breach 15s limit
start_time: float = time.time()


def reset_likelihoods():
    global likehoods
    likehoods = {}


def process_data(imgs: list[ImageData], p: BayesParamCombo):
    for i in range(len(imgs)): # Reducing number of disctinct color levels
        for j in range(len(imgs[i].data)):
            imgs[i].data[j] //= 256 // p.gray_vals + 1

    global max_value
    max_value = p.gray_vals

    if (p.conv_size > 0): # 2D shortening
        for i in range(len(imgs)):
            imgs[i].data = twod_min_select(imgs[i].data, p.conv_size)
    elif (p.len_div > 0): # 1D shortening
        for i in range(len(imgs)):
            new_arr = []
            for j in range(0, len(imgs[i].data) - p.len_div, p.len_div):
                new_arr.append(min(imgs[i].data[j : j + p.len_div]))
            imgs[i].data = np.array(new_arr)


def process_all_data(
    train_images: list[ImageData],
    val_images: list[ImageData],
    test_images: Optional[list[ImageData]],
    p: BayesParamCombo
):
    process_data(train_images, p)
    process_data(val_images, p)
    if test_images is not None:
        process_data(test_images, p)


def train(truth_dict: dict[str, str], train_images: list[ImageData], lap: int):
    global possible_states, likehoods
    possible_states = list(dict.fromkeys(list(truth_dict.values())))

    # Go through each image and updade likelihood "metadictionary" using the values.
    for image in train_images:
        state = image.state
        upsert_key(state, likehoods, {})
        for i in range(len(image.data)):
            upsert_key(i, likehoods[state], {})
            px = image.data[i]
            total = 1 + lap
            for j in range(max_value):
                upsert_key(j, likehoods[state][i], LHoodTableValue(lap, total))
                likehoods[state][i][j].inc_total()
            likehoods[state][i][px].inc_count()


def eval_image(image: ImageData) -> str:
    state_probabilities: dict[str, int] = {}
    for ps in possible_states:
        prob_contribs = np.array(
            [likehoods[ps][i][image.data[i]].get_prob() for i in range(len(image.data))]
        )
        state_probabilities[ps] = np.sum(np.log(prob_contribs))
 
    # a[1] is probability, max(...)[0] is the state
    return max(list(state_probabilities.items()), key=lambda a: a[1])[0]


def tune_hyperparams(
    param_combos: list[BayesParamCombo],
    truth_dict,
    train_images,
    val_images,
):
    max_acc, max_params = 0, None

    for p in param_combos:
        print(time.time() - start_time)
        if (time.time() - start_time > 10): # Keep some time for final training
            break
        train_images_cp, val_images_cp, _ = get_data_copy(train_images, val_images)
        reset_likelihoods()

        process_all_data(train_images_cp, val_images_cp, None, p)
        train(truth_dict, train_images_cp, p.alpha)

        acc = validate_model(truth_dict, val_images_cp, eval_image)
        max_params = p if acc > max_acc else max_params
        max_acc = max(acc, max_acc)
        # print(
        #     f"[{str(p)}] -> Acc: {acc:.3f};; Max: [{str(max_params)}] -> Acc: {max_acc:.3f}"
        # )

    return max_params


def tune_hyperparams_quick(truth_dict, train_images, val_images):
    """
    Online hyperparameter tuning to be used when submitted. Runs just a few
    combinations (on a smaller amount of training data?) (which were
    pre-selected when offline testing) to select the best one for the current data.
    """

    # Can only do 8-ish parameter combinations here
    param_combos: list[BayesParamCombo] = []
    for i in np.arange(0.21, 0.29, 0.04):  # LAP Number
        for j in range(4, 8, 2):  # Number of grayscale values
            for k in range(4, 8, 2): # 1D downscaling
                param_combos.append(BayesParamCombo(i, j, k, 0))
            # for k in range(, 2, 1):  # 2D downscaling
            #     param_combos.append(BayesParamCombo(i, j, 0, k))

    return tune_hyperparams(param_combos, truth_dict, train_images, val_images)


def train_test(
    truth_dict: dict[str, str],
    train_images: list[ImageData],
    val_images: list[ImageData],
    test_images: list[ImageData],
    out_file: str,
    p: Optional[BayesParamCombo],  # Best parameters
):
    """Train the model and evaluate test data."""
    # Get fresh copy of data to process and reset likelihoods table
    train_images_cp, val_images_cp, test_images_cp = get_data_copy(
        train_images, val_images, test_images
    )
    reset_likelihoods()

    process_all_data(train_images_cp, val_images_cp, test_images_cp, p)
    # Train on train+validation data since hyperparameters are fixed at this point.
    # truth_dict contains values for validation data as well.
    train(truth_dict, train_images_cp + val_images_cp, p.alpha)

    # Evaluate (this also saves output file)
    evaluate_data(test_images_cp, out_file, eval_image)


def main():
    parser = setup_arg_parser(method=Method.BAYES)
    args = parser.parse_args()

    print(
        f"Runnning Bayes -> Directories: [Train: {args.train_path},"
        + f"test: {args.test_path}], Output: {args.o}"
    )

    # Load all required data (shared across both classifiers)
    truth_dict, train_images, val_images, test_images = get_structured_data(args)

    # Quickly tune hyperparameters based on data (e.g. 10x10 vs 28x28)...
    best_params = tune_hyperparams_quick(truth_dict, train_images, val_images)
    # Do final training and evaluate test data
    train_test(truth_dict, train_images, val_images, test_images, args.o, best_params)


if __name__ == "__main__":
    main()
