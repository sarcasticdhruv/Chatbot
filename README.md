# Chatbot using PyTorch and NLTK

This repository contains a simple chatbot implementation using PyTorch. The chatbot is trained on the sample intents dataset general use which can be molded accordings to implemetation. The model is a basic LSTM-based sequence-to-sequence architecture.

## Requirements
- Python 3
- PyTorch
- NLTK

## Files

- **model.py**: Defines the neural network architecture (`NeuralNet`) for the chatbot.

- **train.py**: Contains the training code for the chatbot. It uses a dataset defined in `intents.json` to train the neural network.

- **chat.py**: Implements the chat functionality of the trained model. It loads the trained model from `data.pth` and allows users to interact with the chatbot through the command line.

- **start.py**: Entry point for running the chatbot. It checks whether the model has been trained and either trains the model or runs the chat accordingly.

- **nltk_utils.py**: Contains utility functions for tokenization and word stemming using the NLTK library.

## Usage
1. Fork/Clone/Download this repo:

```bash
git clone https://github.com/sarcasticdhruv/Chatbot
```

2. Navigate to the directory:

```bash
cd Chatbot
```

3. Run:

```bash
pip install -r requirements.txt
```

4. Now Run start.py to start the bot:

```bash
python start.py
```
This command will run a python script which check if Model is trained or not, if not it is trained.

5. For training the model again:
```bash
python train.py
python start.py
```

6. And if above didn't work properly:
```bash
python train.py
python chat.py
```

## Code Overview

### Model Architecture
The chatbot model is defined in `model.py` and consists of a simple feedforward neural network implemented using PyTorch. The architecture includes three linear layers with ReLU activation in between. The model predicts the intent of user input without applying an activation function or softmax at the end.

### Training
The training script, `train.py`, processes the intents and patterns defined in `intents.json` to create a dataset. It uses a bag-of-words representation and PyTorch's CrossEntropyLoss for training. The model is trained using the Adam optimizer over a specified number of epochs.

### Chat
The chat functionality is implemented in `chat.py`. It loads the trained model from `data.pth` and allows users to interact with the chatbot through the command line. The input is tokenized and converted into a bag-of-words representation, and the trained model predicts the intent.

### Start
The entry point for running the chatbot is `start.py`. It checks whether the model has been trained and either trains the model using `train.py` or runs the chat using `chat.py`.

### NLTK Utilities
`nltk_utils.py` contains utility functions for tokenization and word stemming using the NLTK library. These functions are used in the data preprocessing pipeline.

## Customization
Feel free to customize the model architecture, hyperparameters, or the training dataset by modifying the respective files (`model.py`, `train.py`, `intents.json`). Experiment with different neural network architectures, training data, or preprocessing techniques to adapt the chatbot to your specific use case.

## Acknowledgments
- The chatbot model is implemented using PyTorch, a powerful deep learning library.
- NLTK (Natural Language Toolkit) is utilized for tokenization and stemming in text preprocessing.
- The project structure is designed for simplicity and ease of understanding.