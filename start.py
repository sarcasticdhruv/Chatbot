import os
import nltk

def is_model_trained():
    # Checks if a file or some other indicator exists to determine if the model has been trained
    return os.path.exists("data.pth")

def train_model():
    nltk.download('punkt')
    print("Training the model...")
    os.system("python train.py")

def run_chat():
    print("Running chat...")
    os.system("python chat.py")

def main():
    if is_model_trained():
        run_chat()
    else:
        train_model()
        run_chat()

if __name__ == "__main__":
    main()
