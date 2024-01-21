# Chatbot using PyTorch and TorchText

This repository contains a simple chatbot implementation using PyTorch and TorchText. The chatbot is trained on the Multi30k dataset for English-German translation. The model is a basic LSTM-based sequence-to-sequence architecture.

## Requirements
- Python 3
- PyTorch
- TorchText
- SpaCy

## Usage
1. Install the required dependencies:

```bash
pip install torch torchtext spacy
```

2. Download the SpaCy English tokenizer:

```bash
python -m spacy download en
```

3. Run the `chatbot.py` script to train the model:

```bash
python chatbot.py
```

4. After training, you can interact with the chatbot by providing input when prompted.

## Code Overview

### Model Architecture
The chatbot model is defined in `chatbot_model.py` and consists of an LSTM-based sequence-to-sequence architecture. The LSTM is followed by a fully connected layer to produce the output.

### Training
The training script, `train.py`, loads the Multi30k dataset, builds the vocabulary, and trains the chatbot model using cross-entropy loss and the Adam optimizer.

### Inference
The `inference.py` script allows interaction with the trained chatbot model. It takes user input, tokenizes it, and generates a response using the trained model.

## Customization
Feel free to customize the model architecture, hyperparameters, or use a different dataset for training to adapt the chatbot to your specific use case.

## Acknowledgments
- The chatbot model is based on the sequence-to-sequence architecture commonly used in machine translation tasks.
- The Multi30k dataset is used for training, and TorchText is employed for data preprocessing.

Please refer to the respective scripts for more details on implementation and usage. If you have any questions or suggestions, feel free to open an issue or submit a pull request.