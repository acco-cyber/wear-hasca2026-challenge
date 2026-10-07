Submissions are evaluated on [macro F1-Score](https://en.wikipedia.org/wiki/F-score#Macro_F1) between the predicted and the ground truth activity label of each sliding window in the test set. The macro F1-score is the arithmetic mean of class-wise F1 scores.

## Submission File
For each ID in the test set, you must predict the activity label for the `target_feature `. The file should contain a header and have the following format:

    id,target_feature
    1,0
    2,10
    3,2
    etc.

For more details see the [Data](https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026/data) section of this challenge