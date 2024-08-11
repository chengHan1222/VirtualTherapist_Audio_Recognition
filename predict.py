import time
import joblib
import librosa
import soundfile
import numpy as np
import noisereduce as nr

# Load model
model = joblib.load('./app/ser/model/RandomForest_20211015_09311.joblib')

def extract_feature(file_name):
    """Extract audio features:
    audio channel should be "mono"!
    (1) chromagram
    (2) mel-scaled spectrogram
    (3) MFCCs
    (4) spectral contrast
    (5) Tonnetz
    """
    with soundfile.SoundFile(file_name) as sound_file:
        # Load voice
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        # X, sample_rate = librosa.load(sound_file)
        
        # Noise reduction
        # X = nr.reduce_noise(y=X, sr=sample_rate)

        # Extract features
        stft = np.abs(librosa.stft(X))
        feature = np.array([])

        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature = np.hstack((feature, mfccs))

        chroma = np.mean(librosa.feature.chroma_stft(y=X, S=stft, sr=sample_rate).T, axis=0)
        feature = np.hstack((feature, chroma))

        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        feature = np.hstack((feature, mel))

        contrast = np.mean(librosa.feature.spectral_contrast(y=X, S=stft, sr=sample_rate).T,axis=0)
        feature = np.hstack((feature, contrast))

        tonnetz = np.mean(librosa.feature.tonnetz(y=X, sr=sample_rate).T,axis=0)
        feature = np.hstack((feature, tonnetz))

    return feature


def predict_emotion_proba(audio):
    """Return probability of 3 emotions
    e.g. {'negative': '0.056369', 'positive': '0.017814', 'neutral': '0.000000'}
    """
    # Extract features
    feature = extract_feature(file_name=audio)

    # Predict
    emotion = model.predict_proba(np.array([feature]))

    # Format result
    formatted_list = ['%.6f' % elem for elem in emotion[0]]
    result = zip(model.classes_, formatted_list)
    return dict(result)


def predict_emotion(audio):
    """Return the most likely emotion
    """
    # Extract features
    feature = extract_feature(file_name=audio)

    # Predict
    emotion = model.predict(np.array([feature]))
    return emotion[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="path to .wav file (audio channel should be \"mono\")",
    )
    parser.add_argument(
        "--predict",
        type=str,
        required=False,
        help="add \"proba\" to predict probability of 5 emotions",
    )

    args = parser.parse_args()

    if args.predict == "proba":
        result = predict_emotion_proba(args.audio)
    else:
        result = predict_emotion(args.audio)
    print(result)


if __name__ == '__main__':
    main()
