Submissions are evaluated on [macro F1-Score](https://en.wikipedia.org/wiki/F-score#Macro_F1) between the predicted and the ground truth activity label of each sliding window in the test set. The macro F1-score is the arithmetic mean of class-wise F1 scores.

The leaderboard provides you with baselines of a classic [DeepConvLSTM](https://www.mdpi.com/1424-8220/16/1/115), [Attend-and-Discriminate](https://dl.acm.org/doi/10.1145/3448083) and [TinyHAR](https://dl.acm.org/doi/10.1145/3544794.3558467) model being trained per-modality (i.e. one model per modality) on the train data without any modifications. 

## Submission File
For each ID in the test set, you must predict the activity label for the `target_feature `. The file should contain a header and have the following format:

    id,target_feature
    1,0
    2,10
    3,2
    etc.

In order to be properly scored, please apply the following label encoding per activity label:

*    `null`: 0,
*    `jogging`: 1,
*    `jogging (rotating arms)`: 2,
*    `jogging (skipping)`: 3,
*    `jogging (sidesteps)`: 4,
*    `jogging (butt-kicks)`: 5,
*    `stretching (triceps)`: 6,
*    `stretching (lunging)`: 7,
*    `stretching (shoulders)`: 8,
*    `stretching (hamstrings)`: 9,
*    `stretching (lumbar rotation)`: 10,
*    `push-ups`: 11,
*    `push-ups (complex)`: 12,
*    `sit-ups`: 13,
*    `sit-ups (complex)`: 14,
*    `burpees`: 15,
*    `lunges`: 16,
*    `lunges (complex)`: 17,
*    `bench-dips`: 18