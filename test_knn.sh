# Train & test data
TRAIN_DATA="./train_data_4"
TEST_DATA="./test_data_4"

CLASS_FILE_NAME="./classification.dsv"

# Run train/test
python ./submission/knn.py $TRAIN_DATA $TEST_DATA

# Get success rate
python ./test_success.py $TEST_DATA/truth.dsv ./classification.dsv