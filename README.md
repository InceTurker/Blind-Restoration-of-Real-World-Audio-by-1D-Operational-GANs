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


![743](https://user-images.githubusercontent.com/117115792/209517383-60b4229a-aeae-48f8-8cc5-2910f020ae3b.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/9d1ac819-31e1-4203-9017-699dcfb9bd7b

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/ce6ccf12-1ae9-45c0-9dff-d8050a14784a

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/5cad6181-eb4d-4dc8-a099-3d4bf9add372

![752](https://user-images.githubusercontent.com/117115792/209517392-1d430f84-4adc-4206-94cb-cf281135ce7b.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/ecfa7334-194a-40dd-b179-d689d7e25a6d

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/ac602293-05a7-41bc-b8ec-5998355f9ed3

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/9b045afb-4616-4fd8-a5dd-c58b0fd9244d

![963](https://user-images.githubusercontent.com/117115792/209517402-17eae4b8-f9ed-4467-aa20-9766376eef48.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/83656ab3-e3e7-4b0a-b519-2283c2c2627a

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/18c64a16-6571-45a3-88c9-f77afbb78400

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/456cd41f-9a2e-4f05-ab25-bc5c91f4b52e

![988](https://user-images.githubusercontent.com/117115792/209517414-a68d633f-e111-49a9-a1d5-2114ae74ef75.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/82af0d4f-0db1-4db4-ac9e-1cde878ed7ce

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/0c91eab8-7c8a-499b-aaff-15511b02c13f

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/a71cc5a5-f4b5-4b50-a240-bd5019dd55c5

![1000](https://user-images.githubusercontent.com/117115792/209517422-11ee15d6-15de-407b-a8c5-db4d6aed2e7f.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/93c04dc4-7f12-4a68-99ed-722cecaed6f0

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/65c1e2f2-36a0-412a-b197-a55995c79c14

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/3439ad13-be86-4d67-a71d-9c86b4672d66

![1007](https://user-images.githubusercontent.com/117115792/209517429-2bd68315-37d0-4cfd-bc89-5bfe2cef5dcd.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/e52bb93b-fde0-4e14-896e-eb37be3bb689

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/813402e2-e90d-49dd-b5c1-c006f33095c0

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/52da2a1a-4463-4d40-88cf-b6d8756953ac

![1008](https://user-images.githubusercontent.com/117115792/209517440-ac2d9068-4b20-4ddd-adb6-735edd9af2cf.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/5945949f-79bb-4976-84e5-028e40baccb8

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/b71de7a4-8bdb-44e3-b071-1e9900b98f37

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/fa690e95-a90f-4f72-a9de-3dee7ab685ee

![1009](https://user-images.githubusercontent.com/117115792/209517444-1562c27b-24cc-436a-b8b7-88cbb5f33b70.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/c196874f-bc84-4f85-a7aa-9a5c5e6648b4

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/eb818033-a107-492e-ac4c-1ec0025256d4

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/f0d6b6c6-c6e9-4701-8c8b-e999c2578547

![1027](https://user-images.githubusercontent.com/117115792/209517448-7494655d-e45b-43cc-85f3-21c1d66ecae1.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/b254b8d0-c6ae-4cd8-8452-229a6f5b0744

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/f2c02f54-30a5-4957-9cbf-b239562903a5

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/a9241767-3eee-4844-997c-d5bf42608424

![1034](https://user-images.githubusercontent.com/117115792/209517457-6de66000-d116-4a0b-b9e8-1ffba4ce5483.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/395de25b-47a3-4497-bc54-c04b2b75fd36

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/9936a857-0bf0-4789-a276-77d94535ad7c

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/ba659da6-9521-4913-87da-9a0494b57192

![1265](https://user-images.githubusercontent.com/117115792/209517470-31c87daf-70ce-494f-b0fc-ff6eb90e1782.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/5eddb17c-4a21-49ca-a18c-2e799e3085ce

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/4f531804-94f3-45d4-ba97-cd4bdee07bef

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/b974137f-8fbb-4a02-98bf-cbd765369bef

![1284](https://user-images.githubusercontent.com/117115792/209517474-e37ac407-ffd6-49a5-b942-83ba7d113ab5.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/1ec98c52-570a-422f-b26b-03229c95341a

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/f324f27e-2114-4772-867c-845bafbd286c

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/eb424221-344f-403f-aa69-7c61ce205efb

![7](https://user-images.githubusercontent.com/117115792/209510044-2fde7e8c-9151-4b79-bc05-202a3ee8b9c2.png)

Clean Signal:


https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/6e01928b-534b-437a-9783-9f7d165e8287

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/3f77b922-465a-417b-8a22-b7a50c5e946e

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/746efc6f-9a8b-4175-a90b-68a26eb998c7

![32](https://user-images.githubusercontent.com/117115792/209510142-75efddbd-483f-4df0-8ded-1e12b39b13a7.png)

Clean Signal:


https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/b3bb00af-8e7b-4d7e-8bec-2a790f62edef

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/f0516978-8d44-4a26-a77e-3f9a610683c1

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/1fe9ec58-a5a0-4767-912f-73f35f9025e9

![30](https://user-images.githubusercontent.com/117115792/209517363-7611fab5-fa90-4df9-9693-c0a449a5c48f.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/d700a0a2-0891-478b-a004-06ade020da92

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/2e3d9554-6ae4-4109-93cb-bd3924746dbd

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/23d41632-4c5d-4f9d-ad51-77fc1aaef461

![500](https://user-images.githubusercontent.com/117115792/209517375-f5a1cc89-c20e-483e-99d5-48b00602c550.png)

Clean Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/5d4ed27b-153d-4ddd-bf95-519c7449ce30

Corrupted Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/97d20b1e-423d-4ef7-8197-0fa54ea99374

Restored Signal:

https://github.com/InceTurker/Blind-Restoration-of-Real-World-Audio-by-1D-Operational-GANs/assets/117115792/b50c03c0-0ff5-4059-b9cf-5244377fcd38


## Citation
If you find this project useful, we would be grateful if you cite this paper：

```http

```
