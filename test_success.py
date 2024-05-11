import argparse

from utils import read_truth_file


def main():
    description = ("Compare two .dsv files")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "t", metavar="filepath", help="path to testing truth data", default="./test_data/truth.dsv"
    )
    parser.add_argument(
        "c", metavar="filepath", help="path to testing classification data", default="./classification.dsv"
    )
    args = parser.parse_args()
    truth = read_truth_file(args.t)
    classif = read_truth_file(args.c)

    good, bad = 0, 0
    for line in truth:
        good += line in classif and truth[line] == classif[line]
        bad += line not in classif or truth[line] != classif[line]
    
    print("Testing data success rate: ", good / (good + bad))

if __name__ == "__main__":
   main()
