# Sealions
Project of the course "Image Analysis and Computer Vision" at Politecnico di Milano year 2017/18

# Dataset generation
Notebooks used to generate the .csv with the coordinates of the patches
- extract_sea_lions_coordinates.ipynb
- extract_empty_patches_coordinates.ipynb

.csv files respectively for train, validation and test of sea lions and background
- sealions_train.csv
- sealions_validation.csv
- sealions_test.csv
- empty_train.csv
- empty_validation.csv
- empty_test.csv

Notebooks used to extract the training patches
- build_dataset_0.ipynb
- build_dataset_1.ipynb
- build_dataset_2.ipynb

# Model building and testing
First assignment
- binary_classifier_0.ipynb

Second assigment
- binary_classifier_1.ipynb
- binary_classifier_1_testing.ipynb

Third assignment
- binary_classifier_2.ipynb
- binary_classifier_2_metrics_history.ipynb
- binary_classifier_2_testing.ipynb

Final assignment
- fully_convolutional_model.ipynb

# Results
First model weights
- /binary_classifier/net_0_weights.h5

Second model weights
- /binary_classifier/net_1_weights.h5

Third model weights and structure with training results
- /binary_classifier/net_2_model.h5
- /binary_classifier/metrics_lr0,0005_epochs60.csv

Fully convolutional network detection samples
- detection.ipynb

.csv with wrong predicted patches
- wrong_predictions.csv

NB In the repository are missing the images provided by Kaggle (except for few samples loaded in 'sample_images' folder) and the extracted patches due to the size and number of the images
Here is the link to download the dataset (the images used ranged from 0 to 1000 except those marked as mismatched in the coordinates extraction) https://www.kaggle.com/c/6116/download/KaggleNOAASeaLions.7z
