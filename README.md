# Speech Emotion Recognition
Recognize 3 emotional states (neutral, positive, and negative) from speech.

# Environment
* python 3.8

```
pip install -r requirements.txt
```

# Dataset

## RAVDESS
> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PloS one, 13(5), e0196391.

* [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)]((https://zenodo.org/record/1188976))
* Speech file (Audio_Speech_Actors_01-24.zip, 215 MB) contains 1440 files: 60 trials per actor x 24 actors = 1440. 
* Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions

## EmoDB
> Burkhardt, F., Paeschke, A., Rolfes, M., Sendlmeier, W. F., & Weiss, B. (2005, September). A database of German emotional speech. In Interspeech (Vol. 5, pp. 1517-1520).

* [EmoDB database](http://www.emodb.bilderbar.info/) is created by the Institute of Communication Science, Technical University, Berlin, Germany.
* 10 professional speakers (5 males and 5 females) participated in data recording.
* The database contains a total of 535 utterances and seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral.

## CMU-MOSEI
> Zadeh, A. B., Liang, P. P., Poria, S., Cambria, E., & Morency, L. P. (2018, July). Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 2236-2246).

* [CMU Multimodal Opinion Sentiment and Emotion Intensity (CMU-MOSEI)](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) dataset is the largest dataset of multimodal sentiment analysis and emotion recognition to date.
* The dataset contains more than 23,500 sentence utterance videos from more than 1000 online YouTube speakers.
* Sentiment label is range from -3 to 3. 
* We take [-3~-1], [-1~1], and [1~3] as Negative, Neutral, ans Positive respectively.

## TESS
> Toronto Emotional Speech Set that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966).

* [Toronto emotional speech set (TESS) Collection](https://tspace.library.utoronto.ca/handle/1807/24487)
* A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).

## ESD
> Zhou, Kun, et al. "[Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset.](https://arxiv.org/abs/2010.14794)" ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.

* This [dataset](https://github.com/HLTSingapore/Emotional-Speech-Data) contains 350 parallel utterances spoken by 10 native Mandarin speakers, and 10 English speakers with 5 emotional states (neutral, happy, angry, sad and surprise).
* Only using Mandarin part.
* 0001~0010 under `mandarin_audio/`

# Result
|  Dataset  | Test Data (25% of total data) | Accuracy(mean of 5 cross validation) |
|:---------:|:-----------------------------:|:------------------------------------:|
|  RAVDESS  |              359              |                0.7019                |
|   EmoDB   |              134              |                0.7985                |
| CMU-MOSEI |              344              |                0.7005                |
|   TESS    |              700              |                0.9957                |
|    ESD    |             4375              |                0.9618                |

* We map different emotions into only neutral, positive, and negative
* Total accuracy = 93.11%

# Train
Save model under `model/`.
```
python src/train.py
```

# Predict

:warning: audio channel should be \"**mono**\"

1. return the most likely emotion
```
python src/predict.py --audio path/to/audio.wav
```

2. return probability of 5 emotions
```
python src/predict.py --audio path/to/audio.wav --predict proba
```
