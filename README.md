# Dog vs Cat Image Classification Using Deep Learning

## Project Overview

This project develops and evaluates deep learning models for binary image classification to distinguish between dogs and cats. Using TensorFlow and Keras, it explores different approaches to image classification, including a custom Convolutional Neural Network (CNN), transfer learning through feature extraction, and full fine-tuning of pretrained CNN architectures.

The project also investigates data augmentation, regularization, early stopping, model performance comparison, and error analysis to understand how different deep learning approaches perform on unseen images.

## Dataset

**Dataset:** Dog vs Cat Image Dataset

**Source:** [GitHub – Dog Cat Full Dataset](https://github.com/laxmimerit/dog-cat-full-dataset)

The dataset contains labeled images belonging to two categories:

- Cats
- Dogs

### Dataset Structure

```text
dog-cat-full-dataset/
│
└── data/
    ├── train/
    │   ├── cats/
    │   └── dogs/
    │
    └── test/
        ├── cats/
        └── dogs/
```

Images are loaded using TensorFlow's `image_dataset_from_directory()` utility and resized to 224 × 224 pixels.

## Project Objectives

- Prepare and preprocess image datasets for deep learning.
- Create training, validation, and test datasets.
- Apply image normalization and data augmentation.
- Build a custom CNN for binary image classification.
- Implement transfer learning using pretrained CNN architectures.
- Perform full fine-tuning of a pretrained model.
- Apply early stopping to reduce overfitting.
- Compare the performance and computational efficiency of different models.
- Evaluate classification performance using multiple metrics.
- Perform error analysis using classification reports, confusion matrices, and misclassified images.

## Methodology

### 1. Dataset Preparation and Preprocessing

Prepare the image dataset using TensorFlow/Keras.

Tasks include:
- Download and load the dataset.
- Split the training data into 80% training and 20% validation subsets.
- Resize images to 224 × 224 pixels.
- Normalize pixel values to the range [0, 1].
- Inspect class names and dataset sizes.
- Optimize dataset loading using `prefetch()`.

### 2. Data Augmentation

Implement an image augmentation pipeline using TensorFlow preprocessing layers.

Augmentation techniques include:
- Random Flip
- Random Rotation
- Random Zoom
- Additional augmentation, such as Random Contrast

Augmentation is applied only during training to introduce image variation and improve the model's ability to generalize.

Original and augmented image samples are visualized to examine the effects of these transformations.

### 3. Custom CNN Model

Build a Convolutional Neural Network from scratch for binary image classification.

The architecture includes:
- Multiple convolutional blocks.
- MaxPooling layers.
- Batch Normalization.
- Dropout regularization.
- Dense classification layers.
- A Sigmoid output layer for binary classification.

The model is compiled using an appropriate optimizer, binary classification loss function, and evaluation metrics.

An EarlyStopping callback monitors validation loss and restores the best model weights to help prevent overfitting.

Training and validation accuracy and loss are plotted to examine model performance.

### 4. Transfer Learning: Feature Extraction

Apply transfer learning using a pretrained CNN architecture, such as MobileNetV2, EfficientNetB0, ResNet50, or DenseNet121.

The feature extraction approach includes:
- Loading pretrained ImageNet weights.
- Removing the original classification head.
- Freezing the pretrained backbone.
- Adding a new classification head for dog and cat prediction.
- Training only the newly added classifier layers.

This approach uses pretrained visual representations while keeping the backbone weights unchanged.

### 5. Full Fine-Tuning

Use a different pretrained CNN architecture from the one selected for feature extraction.

The fine-tuning process includes:
- Loading the pretrained model.
- Replacing the original classification head.
- Unfreezing the pretrained layers.
- Training the network on the dog and cat dataset.
- Using EarlyStopping to monitor validation performance.

Unlike feature extraction, full fine-tuning allows pretrained weights to adapt to the target dataset.

### 6. Model Comparison

Compare the performance of three approaches:

1. Custom CNN
2. Transfer Learning (Feature Extraction)
3. Transfer Learning (Full Fine-Tuning)

Evaluate the models using the following metrics:

| Metric | Description |
|---|---|
| Total Parameters | Number of model parameters |
| Training Loss | Loss on the training dataset |
| Training Accuracy | Accuracy on the training dataset |
| Validation Loss | Loss on the validation dataset |
| Validation Accuracy | Accuracy on the validation dataset |
| Test Loss | Loss on the test dataset |
| Test Accuracy | Overall test classification accuracy |
| Test Precision | Proportion of predicted positive cases that are correct |
| Test Recall | Proportion of actual positive cases correctly identified |
| Test F1 Score | Harmonic mean of precision and recall |

The results are organized into a comparison table to identify differences in predictive performance and computational requirements.

### 7. Error Analysis

Perform detailed error analysis to understand the classification mistakes made by the models.

The analysis includes:
- Generating predictions on the test dataset.
- Displaying a classification report.
- Creating a confusion matrix.
- Visualizing at least nine incorrectly classified images.
- Showing actual labels, predicted labels, and prediction confidence for each misclassified image.

This helps identify challenging examples and understand where the models struggle.

## Model Evaluation and Results

Complete this section with the actual experimental results.

Include:
- Training and validation accuracy/loss curves.
- Model comparison table.
- Test accuracy, precision, recall, and F1 scores.
- Confusion matrix for the selected model.
- Examples of incorrectly classified images.
- Observations about the effectiveness of feature extraction versus full fine-tuning.

Also discuss:
- Which model achieved the best performance?
- Which model was most computationally efficient?
- Which model would be most appropriate for deployment, considering accuracy, inference cost, and model complexity?

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Project Structure

```text
dog-cat-image-classifier/
│
├── Dog_Cat_Image_Classifier.ipynb
├── README.md
└── models/
    └── best_model.keras
```

The notebook contains dataset preparation, augmentation, model development, training, evaluation, and error analysis. The trained model can be saved separately if needed.

## How to Run

1. Open the notebook in Google Colab.
2. Clone or download the dataset from the provided GitHub repository.
3. Install the required dependencies if necessary.
4. Run the notebook cells in sequence.
5. Train the custom CNN and pretrained models.
6. Review the evaluation metrics, visualizations, and error analysis.

A GPU-enabled runtime is recommended for training and fine-tuning pretrained CNN architectures.

## Conclusion

This project demonstrates a complete deep learning workflow for image classification, from dataset preparation and augmentation to model development and evaluation.

By comparing a custom CNN with transfer learning and full fine-tuning approaches, the project explores how pretrained visual representations can improve image classification performance and how different training strategies affect model efficiency and generalization.

## Author

Nadiya Nowshin
