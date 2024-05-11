# This script sorts data into training and testing folders,
# which can then be used as inputs for the program.
# IT SHOULD NOT MODIFY THE DATA IN ANY WAY.

from math import floor
import os
import random
import shutil

from utils import read_truth_file

curr_path = os.path.dirname(__file__)

train_path, test_path = None, None


def re_create_dirs() -> tuple[str, str]:
    train_path = os.path.join(curr_path, "train_data_4")
    test_path = os.path.join(curr_path, "test_data_4")
    if os.path.exists(train_path):
        shutil.rmtree(train_path)

    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    os.makedirs(train_path)
    os.makedirs(test_path)
    return train_path, test_path


def parse_data_source_folder(source_name: str, f_num: int):
    src_path = os.path.join(curr_path, source_name)
    dir_list = os.listdir(src_path)

    # This line of code is not used because we get information about 
    # images from the truth.dsv file. That will not be possible when working
    # with real test data!
    # images = list(filter(lambda a: ".png" in a, dir_list))

    truth = list(filter(lambda a: ".dsv" in a, dir_list))[0]
    truth_dict = read_truth_file(os.path.join(src_path, truth))

    # Sort it randomly, but optionally controlled through random.seed()
    truth_list = list(truth_dict.items())
    random.shuffle(truth_list)

    total_num = len(truth_list)

    # Create truth files in training and testing directories...
    # Note: The testing truth.dsv file is for evaluation purposes only.
    # The code cannot actually EXPECT it

    train_truth, test_truth = [], []

    for i in range(total_num):
        if i % total_num < 0.8 * total_num:
            train_truth.append(f"{f_num}{truth_list[i][0]}:{truth_list[i][1]}")
        else:
            test_truth.append(f"{f_num}{truth_list[i][0]}:{truth_list[i][1]}")

    with open(os.path.join(train_path, "truth.dsv"), "a") as f:
        f.write("\n".join(train_truth))
        f.write("\n")

    with open(os.path.join(test_path, "truth.dsv"), "a") as f:
        f.write("\n".join(test_truth))
        f.write("\n")

    for train_t in train_truth:
        f_name = train_t.split(":")[0]
        shutil.copy(
            os.path.join(src_path, f_name[1:]), os.path.join(train_path, f_name)
        )

    for test_t in test_truth:
        f_name = test_t.split(":")[0]
        shutil.copy(os.path.join(src_path, f_name[1:]), os.path.join(test_path, f_name))

    # TODO: This really SHOULD BE implemented (making sure there are same numbers
    # of all classes in training data....), but eh...
    # Count of each class in training data... want to keep this consistent
    # count_in_training: dict[str, int] = {}

    # for k, v in truth_list:
    #     if k not in count_in_training:
    #         count_in_training[k] = 0
    #     if count_in_training[k] <

    # print(images, truth_dict)


def get_list_of_all_data() -> list[str]:
    # source_names = ["train_700_28"]
    # source_names = ["train_1000_28"]
    # source_names = ["train_1000_10"]
    source_names = ["train_1000_28", "train_700_28"]

    for i in range(len(source_names)):
        parse_data_source_folder(source_names[i], i)


def main():
    print("Creating data split...")

    print("1. Recreating directories...")
    global train_path, test_path
    train_path, test_path = re_create_dirs()

    print("2. Copying files...")
    get_list_of_all_data()


if __name__ == "__main__":
    main()
