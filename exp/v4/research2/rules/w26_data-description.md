⚠️ Update – April 26, 2025: A bug in the video feature generation has been fixed. Please redownload the test data. Sample ordering is unchanged; prior submissions without video features are unaffected.

## Files

*   **train** – raw inertial data and frame-wise VideoMAEv2-Base features of the WEAR dataset, split by participant. Filenames ending in *_2.csv indicate a repeated recording (see [paper](https://arxiv.org/abs/2304.05088v4)). Note the difference in sampling rates: inertial = 50 Hz, video = 30 FPS.
* **test** – all test split files (see below for details)
* `sample_submission.csv` – a sample submission file in the correct format
* `participant_meta.txt` – supplemental metadata about participants and recording conditions of the test set (for train set info, see the [WEAR paper supplementary material](https://github.com/mariusbock/wear/blob/main/supplementary_material.pdf))

## Training Data

Each inertial `.csv` file contains the following columns:
*   `sbj_id` - Unique participant id (0–21)
*   `right_arm_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, right wrist
*   `right_leg_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, right leg
*   `left_leg_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, left leg
*   `left_arm_acc_{x,y,z}` - 3-axis acceleration at 50 Hz, left arm
*   `label ` - Activity label

Each `.npy` file contains pre-extracted video features for the corresponding participant's egocentric video. Features were extracted using [VideoMAEv2-Base](https://huggingface.co/OpenGVLab/VideoMAEv2-Base), a transformer-based video understanding model pretrained on large-scale video data. Participants can use these features as a compact, high-level representation of the visual content.
Features were extracted at 30 FPS with frames resized to `224×224` (matching the format of the pretraining data). For each frame `i`, the model takes a clip of `16` frames (frame i, 8 past frames, and 7 future frames) as input and outputs a single `768`-dimensional feature vector. Each row `i` in the `.npy` file corresponds to this feature vector for frame `i`.

## Test Data

The test set consists of `N=12,234` pre-windowed samples across three files:

* `test_inertial_data.npy`: Inertial windows at 50 Hz; shape `(N, 50, 3)` per sensor location
* `test_videomae_data.npy`: VideoMAEv2 features at 30 FPS; shape `(N, 15, 768)`. Windows contain `15` feature vectors instead of the `30` to avoid frames near the boundaries of a window to leak information about past/ future context not part of the 1 second windows.
* `test_meta_data.csv`: Window `id`, `subject_id`, and `inertial_sensor_location` for each window

All three files share the same ordering, defined by the `id` column in `test_meta_data.csv`.

## Columns Submission File & Label Encoding
To submit a solution please create a `.csv` with columns `id` and `target_value` (predicted activity label) for all test windows, using the following mapping:

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