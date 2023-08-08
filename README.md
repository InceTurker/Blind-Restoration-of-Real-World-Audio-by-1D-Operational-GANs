# Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs


# Project Description

Objective: Despite numerous studies proposed for audio restoration in the literature, most of them focused on an isolated restoration problem such as denoising or dereverberation, ignoring the other artifacts. Moreover, assuming a limited number of signal-to-distortion ratio (SDR) levels is a common practice. However, real-world audio is often corrupted by a blend of artifacts such as reverberation, sensor noise, and background audio mixture with varying types, severities, and duration. In this study, we propose a novel approach for blind restoration of real-world audio signals by Operational Generative Adversarial Networks (Op-GANs) with temporal and spectral objective metrics to enhance the quality of restored audio signal regardless of the type and severity of each artifact corrupting it. Methods: 1D Operational-GANs are used with the generative neuron model optimized for blind restoration of any corrupted audio signal. Results: The proposed approach has been evaluated extensively over the benchmark TIMIT-RAR (speech) and GTZAN-RAR (non-speech) datasets corrupted with a random blend of artifacts each with a random severity to mimic real-world audio signals. Average SDR improvements over 7.2 dB and 4.9 dB are achieved, respectively. Significance: This is a pioneer study in blind audio restoration with the unique capability of direct (time-domain) restoration of real-world audio whilst achieving an unprecedented level of performance for a wide SDR range and artifact types. Conclusion: 1D Op-GANs can achieve robust and computationally effective real-world audio restoration with an elegant performance level.
[Paper Link](https://arxiv.org/abs/2212.14618)

![image](https://user-images.githubusercontent.com/117115792/209479770-85f967b7-91f9-42f6-af34-08a3412bba1f.png)

## Real-World Audio Dataset 


![image](https://user-images.githubusercontent.com/117115792/209479487-75c1f71b-cf0b-46b3-a60a-c282a856244f.png)

- The proposed formation of both benchmark datasets generated in this study to mimic real-world corrupted audio clips is illustrated in Figure.  
To accomplish this aim, the outputs of randomly selected degradation sources are randomly (~U [0,1]) weighted before corrupting the clean target audio. The two artifacts (AWGN and background mixture) are additive while a linear convolution is applied for reverberation. Therefore, linear (random) weights can be used to control the weights of each artifact type selected. In addition to the random blend of all artifacts, we manually created single artifact cases (only one of is turned on) to be included in the final datasets, which may correspond to the scenarios where only background mixture or reverberation exist. 
 - For the evaluation of speech restoration, a total of 2703 clean data samples are
taken from the [TIMIT Corpus Dataset](https://catalog.ldc.upenn.edu/LDC93s1), which contains recordings of different speakers from 8 major dialects of American English each reading phonetically rich sentences. Each utterance is a 2-second-long (32000 samples) segment with a sampling rate of 16 kHz. For the
training and validation sets, 2000 randomly selected data samples are input to the real-world corrupted audio generation setup. The
final train set includes 1500 samples from the blend of all artifacts as well
as 500 samples per single artifact case, which adds up to a total of 3000 data
samples. Note for each single artifact case samples are selected as
non-overlapping groups (of 500 samples) from randomly selected 1500 train
samples. Similarly, 500 and 703 randomly selected utterances from the remaining
data are used to form the independent validation and test sets, which includes a
total of 1000 and 1453 data samples, respectively. This benchmark dataset that
can henceforth be used for real-world audio restoration is named TIMIT-RAR. 

- Similarly, for the evaluation of non-speech audio restoration, approximately
1.45-second-long segments (32000 samples with a sampling rate of 22050 Hz) from
the classical and jazz music recordings of the [GTZAN Music dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) are used. The
final train set includes 1750 samples from the blend of all artifacts as well
as 500 samples (as non-overlapping groups) per single artifact cases, which
adds up to a total of 3250 data samples. Similarly, 500 and 830 randomly
selected utterances from the remaining data are used to form the independent
validation and test sets, which includes a total of 1000 and 1660 data samples,
respectively. This benchmark dataset that can henceforth be used for real-world
audio restoration is named GTZAN-RAR. The final train, validation, and test
data compositions of both datasets are given in Table 1.

- [TIMIT-RAR Dataset](http://2020.icbeb.org/CSPC2020) and [GTZAN-RAR Dataset](http://2020.icbeb.org/CSPC2020) can be downloaded from given links.

## Run

#### Train
- Download [train]([http://2020.icbeb.org/CSPC2020](https://drive.google.com/drive/folders/1TcmZr9pKsFGgqCR1ubKOnCjeAAAXBs7-?usp=drive_link)), [validation]([http://2020.icbeb.org/CSPC2020](https://drive.google.com/drive/folders/1ZTiAqGnEM0PTRtU390sD5PCfBO6bb_Gp?usp=drive_link)) and [test]([http://2020.icbeb.org/CSPC2020](https://drive.google.com/drive/folders/1tvwakD3zrRCL90clUGobC80Wwr2pvXTr?usp=drive_link)) files to tmats, vmats and temats folders respectively.
- Normalize the data
```http
  python audio_norm.py
```
- Start training
  
```http
  python Supervised_GAN.py
```

- Evaluate the model. You can use pre-trained networks parameters for [First-Pass]([http://2020.icbeb.org/CSPC2020](https://drive.google.com/drive/folders/1YrxhbKjCPvUxkZw6HtwVlqAupEtcaU6a?usp=drive_link)) and [Second-Pass]([http://2020.icbeb.org/CSPC2020](https://drive.google.com/drive/folders/1mrD8BaNqLvuNyZKOiRWdu5pRCJ4mrvWC?usp=drive_link))
  
```http
  python test.py
```

- Calculate the performance metrics.
  
```http
  python stoi_calc.py
```

## Prerequisites
- Pyton 3
- Pytorch
- [FastONN](https://github.com/junaidmalik09/fastonn) 


  
## Results

![30](https://user-images.githubusercontent.com/117115792/209517363-7611fab5-fa90-4df9-9693-c0a449a5c48f.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/31cff137-7147-438b-b74a-3d03d0e040dd

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/faf2ab6b-8397-4a0c-a908-4aae4398bcaf

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/e1c6052a-bce8-44aa-9111-f42c50fa94a7


![500](https://user-images.githubusercontent.com/117115792/209517375-f5a1cc89-c20e-483e-99d5-48b00602c550.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/dc45c71e-ae4f-468d-a301-882fc12e5bb9

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/015e3fa1-e942-4047-95b3-d12f0cdf0c6b

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/72356f6c-7215-4ef8-9638-ad368e2c7edc

![743](https://user-images.githubusercontent.com/117115792/209517383-60b4229a-aeae-48f8-8cc5-2910f020ae3b.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/9b93bbe5-b252-4999-a211-7a638592d76d

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/20db1a5b-7fc4-45f7-a870-747212d55d3c

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/2f382204-7595-46e1-bb58-95a43bb270f9

![752](https://user-images.githubusercontent.com/117115792/209517392-1d430f84-4adc-4206-94cb-cf281135ce7b.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![963](https://user-images.githubusercontent.com/117115792/209517402-17eae4b8-f9ed-4467-aa20-9766376eef48.png)

Clean Signal:




Corrupted Signal:


Restored Signal:


![988](https://user-images.githubusercontent.com/117115792/209517414-a68d633f-e111-49a9-a1d5-2114ae74ef75.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1000](https://user-images.githubusercontent.com/117115792/209517422-11ee15d6-15de-407b-a8c5-db4d6aed2e7f.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1007](https://user-images.githubusercontent.com/117115792/209517429-2bd68315-37d0-4cfd-bc89-5bfe2cef5dcd.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1008](https://user-images.githubusercontent.com/117115792/209517440-ac2d9068-4b20-4ddd-adb6-735edd9af2cf.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1009](https://user-images.githubusercontent.com/117115792/209517444-1562c27b-24cc-436a-b8b7-88cbb5f33b70.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1027](https://user-images.githubusercontent.com/117115792/209517448-7494655d-e45b-43cc-85f3-21c1d66ecae1.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1034](https://user-images.githubusercontent.com/117115792/209517457-6de66000-d116-4a0b-b9e8-1ffba4ce5483.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1265](https://user-images.githubusercontent.com/117115792/209517470-31c87daf-70ce-494f-b0fc-ff6eb90e1782.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![1284](https://user-images.githubusercontent.com/117115792/209517474-e37ac407-ffd6-49a5-b942-83ba7d113ab5.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


![7](https://user-images.githubusercontent.com/117115792/209510044-2fde7e8c-9151-4b79-bc05-202a3ee8b9c2.png)

Clean Signal:


Corrupted Signal:


Restored Signal:



![32](https://user-images.githubusercontent.com/117115792/209510142-75efddbd-483f-4df0-8ded-1e12b39b13a7.png)

Clean Signal:


Corrupted Signal:


Restored Signal:


## Citation
If you find this project useful, we would be grateful if you cite this paper：

```http

```
