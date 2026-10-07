## Files

*   **train** - folder containing the inertial-data files of the original WEAR dataset. Data is split across participants with the filename being the unique participant id. Filenames ending in `*_2.csv` indicate the repeated of a specific participant (see [paper](https://arxiv.org/abs/2304.05088v4) for details) 
*   **test.csv** - the test data which is to be predicted. See below for details.
*   **sample_submission.csv** - a sample submission file in the correct format
*   **meta_data.csv** - supplemental information about the participants and recording conditions of the test set (for information on the train set, please see the supplementary material of the [WEAR paper](https://github.com/mariusbock/wear/blob/main/supplementary_material.pdf))

## Columns (Training Data)

Each training data '.csv'-file follows the same structure.  
*   `sbj_id` - unique participant id (from 0 to 21)
*   `right_arm_acc_x` - x-axis acceleration data sampled at 50Hz captured on the right wrist of a participant
*   `right_arm_acc_y` - y-axis acceleration data sampled at 50Hz captured on the right wrist of a participant
*   `right_arm_acc_z` - z-axis acceleration data sampled at 50Hz captured on the right wrist of a participant
*   `right_leg_acc_x` - x-axis acceleration data sampled at 50Hz captured on the right ankle of a participant
*   `right_leg_acc_y` - y-axis acceleration data sampled at 50Hz captured on the right ankle of a participant
*   `right_leg_acc_z` - z-axis acceleration data sampled at 50Hz captured on the right ankle of a participant
*   `left_leg_acc_x` - x-axis acceleration data sampled at 50Hz captured on the left ankle of a participant
*   `left_leg_acc_y` - y-axis acceleration data sampled at 50Hz captured on the left ankle of a participant
*   `left_leg_acc_z` - z-axis acceleration data sampled at 50Hz captured on the left ankle of a participant
*   `left_leg_acc_x` - x-axis acceleration data sampled at 50Hz captured on the left wrist of a participant
*   `left_leg_acc_y` - y-axis acceleration data sampled at 50Hz captured on the left wrist of a participant
*   `left_leg_acc_z` - z-axis acceleration data sampled at 50Hz captured on the left wrist of a participant
*   `label ` - activity label of data record

## Columns (Test Data)

The `test` data is provided in a windowed format. Each row in the test data represents a sliding window of 1 second (i.e. 50 records) of a specific body location and a specific test participant. Goal is to predict the activity label of the window.

*   `id` - unique test record
*   `sbj_id` - unique test participant id (from 22 to 25)
*   `sensor_location` - body location at which the acceleration data was collected (either `right_arm`, `right_leg`, `left_arm` or `left_leg`)
*   `x_axis` - x-axis acceleration measurements (list of 50 records = 1 second worth of data)
*   `y_axis` - y-axis acceleration measurements (list of 50 records = 1 second worth of data)
*   `z_axis` - z-axis acceleration measurements (list of 50 records = 1 second worth of data)

## Columns Submission File & Label Encoding

In order to submit your solution please create a `.csv` containing the ´id` and `target_value`, i.e. predicted activity label of all windows present in the `test.csv` file.

The `label`-column found within the training data can be one of the 19 activity labels present in the WEAR dataset. In order to have your test submission properly scored, please encode the activity labels as follows:

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