import os
import glob
import time
import joblib
import librosa
import soundfile
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        feature = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            feature = np.hstack((feature, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            feature = np.hstack((feature, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            feature = np.hstack((feature, mel))
    return feature

# Emotions
Emotions = ['Neutral', 'Happy', 'Angry', 'Sad', 'Surprise']

# Load all audio files and extract features
def load_data():
    x, y = [],[]
    
    for i in tqdm(range(1, 11), desc="Speaker"): # speaker: 0001 ~ 00010
        index = str(i).zfill(4) # padding i to 4 digits, e.g. 1 -> 0001
        for emotion in Emotions: # every speaker has 5 emotion audio folders
            for folder in tqdm(['evaluation', 'test', 'train'], desc=emotion):
                for file in glob.glob("mandarin_audio/"+index+"/"+emotion+"/"+folder+"/*.wav"):
                    file_name = os.path.basename(file)
                    feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
                    x.append(feature)
                    y.append(emotion)

    return train_test_split(np.array(x), y, test_size=0.25, random_state=66)

# Split the dataset
print("Load and extract features...")
x_train, x_test, y_train, y_test = load_data()

# Get the shape of the training and testing datasets
print(("Train, Test: ", x_train.shape[0], x_test.shape[0]))

# Get the number of features extracted
# length of feature vector (1-d array)
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
# MLPClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

# Train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate accuracy, precesion, recall and f1 score
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
precision = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0, average='weighted')
recall = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
f1score = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1score)

# Save model
nowTime = int(time.time()) # get current time
struct_time = time.localtime(nowTime)
timeString = time.strftime("%Y_%m_%d_%I_%M", struct_time)
joblib.dump(model, 'model/'+ timeString +'.joblib')
