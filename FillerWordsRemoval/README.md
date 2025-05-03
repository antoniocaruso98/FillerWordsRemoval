# Filler Words Removal Project

This project aims to develop a machine learning model for the detection and removal of filler words in audio recordings. The model utilizes a two-headed ResNet architecture for both classification of filler words and regression of bounding box coordinates.

## Project Structure

```
FillerWordsRemoval
├── src
│   ├── models
│   │   └── two_headed_resnet.py  # Defines the two-headed ResNet architecture
│   ├── data
│   │   └── dataset.py             # Dataset class for loading and preprocessing audio data
│   ├── training
│   │   ├── train.py                # Training loop for the model
│   │   └── evaluate.py             # Evaluation logic for model performance
│   ├── utils
│   │   └── helpers.py              # Utility functions for various tasks
│   └── main.py                     # Entry point for the application
├── pretrained_weights
│   └── audio_dataset_weights.pth    # Pre-trained weights for the model
├── requirements.txt                 # Project dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Files to ignore in version control
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd FillerWordsRemoval
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Weights**:
   Place the pre-trained weights file `audio_dataset_weights.pth` in the `pretrained_weights` directory.

## Usage Guidelines

1. **Training the Model**:
   To train the model, run the following command:
   ```
   python src/main.py
   ```

2. **Evaluating the Model**:
   The evaluation will be performed automatically after training. You can also run the evaluation separately by modifying the `main.py` file.

## Model Description

The model is based on a two-headed ResNet architecture:
- **Classification Head**: Predicts the presence of filler words.
- **Bounding Box Regression Head**: Outputs bounding box coordinates for the detected filler words.

## Dataset

The dataset consists of audio recordings annotated with filler words. The `dataset.py` file handles loading and preprocessing of this data, including data augmentation techniques to improve model robustness.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.