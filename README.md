# Breast Cancer Ultrasound Classification via Fine-Tuned ConvNeXt
An end-to-end, explainable AI diagnostic tool designed to classify breast ultrasound images into three categories: Normal, Benign, and Malignant. This project leverages state-of-the-art transfer learning (ConvNeXt), automated hyperparameter tuning (Optuna), and advanced data augmentation to assist radiologists with highly accurate, consistent, and interpretable automated screening.

🚀 Key Features
State-of-the-Art Architecture: Utilizes the ConvNeXt-Tiny architecture, modernized for high-performance computer vision tasks, achieving superior feature extraction compared to traditional CNNs.

Automated Hyperparameter Optimization: Implements Optuna for Bayesian optimization, mathematically isolating the optimal Learning Rate, Weight Decay, and Dropout probability to prevent overfitting and maximize validation accuracy.

Class Imbalance Handling: Integrates dynamically computed Class Weights within the Cross-Entropy Loss function to heavily penalize misclassifications of the minority "Malignant" class, drastically improving clinical sensitivity.

Robust Data Augmentation: Employs PyTorch's v2 transforms (random rotations, flips, color jitter, and antialiased resizing) to artificially expand the dataset and improve model generalization.

Explainable AI (XAI) [Upcoming]: Integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to generate visual heatmaps, highlighting the exact physiological regions driving the model's predictions to build trust with medical professionals.

📊 Dataset
This project utilizes the Breast Ultrasound Images Dataset (BUSI).

Source: Kaggle (sabahesaraki/breast-ultrasound-images-dataset)

Format: 2D Ultrasound Images (PNG) alongside ground truth masks.

Classes: * normal (133 images)

benign (437 images)

malignant (210 images)

🛠️ Tech Stack & Dependencies
Framework: PyTorch (torch, torchvision, torchaudio)

Optimization: Optuna (Bayesian Hyperparameter Tuning)

Data Processing: Pandas, NumPy, Scikit-Learn (Class Weights & Splitting)

Image Processing: Pillow (PIL)

Environment: Google Colab / Jupyter Notebook

⚙️ Pipeline Architecture
Ingestion & Preprocessing: Direct automated download via Kaggle API. Images are resized to 224x224, converted to RGB (3-channel), and normalized using ImageNet standards.

Dataset Splitting: 80/20 Train-Test split utilizing distinct transformations (heavy augmentation for training, clean normalization for testing) via PyTorch Subset and DataLoader.

Tuning Phase: Optuna conducts rapid 5-epoch trials to dynamically search the hyperparameter space.

Training Phase: The model is trained over 30 epochs using the AdamW optimizer and a CosineAnnealingLR scheduler. The model's weights are checkpointed whenever validation accuracy peaks.

Inference & Visualization: Unseen scans are passed through the frozen model to output class probabilities and ultimate diagnostic predictions.

💻 How to Run
1. Prerequisites
Ensure you have a kaggle.json API token from your Kaggle account settings.

2. Installation
Install the required libraries:

Bash
pip install torch torchvision optuna scikit-learn pandas numpy kaggle
3. Execution (Colab/Jupyter)
The project is structured into modular execution cells:

Setup & Download: Authenticate Kaggle and download the BUSI dataset.

Data Preparation: Initialize the v2 transforms, load the ImageFolder, and create the DataLoaders.

Hyperparameter Tuning: Run the Optuna study to find the optimal Dropout, LR, and Weight Decay.

Final Training: Execute the 30-epoch training loop using the optimized parameters. The best model will be saved as convnext_best_mammography_model.pth.

Inference: Pass new image paths to the predict_mammogram() function to view the AI's diagnostic confidence scores.

📈 Results
Acheived excellent validation accuracy of about 95.9%

Loss Optimization: Used CrossEntropyLoss for determining Loss rate

Note: The model prioritizes Sensitivity (Recall) for the Malignant class due to the high clinical cost of false negatives.
