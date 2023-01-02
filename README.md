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
taken from the [TIMIT Corpus Dataset](https://catalog.ldc.upenn.edu/LDC93s1),which contains recordings of different speakers from 8 major dialects of American English each reading phonetically rich sentences. Each utterance is a 2-second-long (32000 samples) segment with a sampling rate of 16 kHz. For the
training and validation sets, 2000 randomly selected data samples are input to the real-world corrupted audio generation setup. The
final train set includes 1500 samples from the blend of all artifacts as well
as 500 samples per single artifact case, which adds up to a total of 3000 data
samples. Note for each single artifact case samples are selected as
non-overlapping groups (of 500 samples) from randomly selected 1500 train
samples. Similarly, 500 and 703 randomly selected utterances from the remaining
data are used to form the independent validation and test sets, which includes a
total of 1000 and 1453 data samples, respectively. This benchmark dataset that
can henceforth be used for real-world audio restoration is named TIMIT-RAR. 

- Similarly, forthe evaluation of non-speech audio restoration, approximately
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
- Will be updated.
```http
  python 1D_Self_Operational_CycleGAN.py
```

## Prerequisites
- Pyton 3
- Pytorch
- [FastONN](https://github.com/junaidmalik09/fastonn) 


  
## Results

![30](https://user-images.githubusercontent.com/117115792/209517363-7611fab5-fa90-4df9-9693-c0a449a5c48f.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/209534081-7136aec8-a14e-4e15-9c7c-ba4ea0b0d29a.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/209534102-c502e4d8-2956-4153-aad5-0c37cabf3c4e.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/209534109-b78c0e9e-a744-43e6-abc0-482b708a1ac4.mp4

![500](https://user-images.githubusercontent.com/117115792/209517375-f5a1cc89-c20e-483e-99d5-48b00602c550.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/209534135-2f15b4e9-a361-4f04-b976-97c271288f42.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/209534148-d412493b-3844-4448-81e1-4bd109bf4be6.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/209534158-b4545f1e-5d16-4e58-8a67-8b398fd6cdf4.mp4

![743](https://user-images.githubusercontent.com/117115792/209517383-60b4229a-aeae-48f8-8cc5-2910f020ae3b.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/209534185-c9727d86-c9b6-405d-a274-a61383d55d60.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/209534196-75d3da2e-1769-4fb7-bf38-c15fbf0f5c94.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/209534213-e19f88c3-38a2-4bc8-9ed7-426f4a692880.mp4

![752](https://user-images.githubusercontent.com/117115792/209517392-1d430f84-4adc-4206-94cb-cf281135ce7b.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/209534247-5a7a98cc-de28-4467-8a38-1f0017ee4c43.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/209534260-77ad666b-3913-4da6-b97b-735477c7d3c7.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/209534282-236780dd-eb95-4206-8eea-4158e7d0ca42.mp4

![963](https://user-images.githubusercontent.com/117115792/209517402-17eae4b8-f9ed-4467-aa20-9766376eef48.png)

Clean Signal:


https://user-images.githubusercontent.com/117115792/210251819-46b4efdc-0a21-48a8-bc0e-03e8ec62d868.mp4


Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210251827-7a23dabb-f70b-4eba-84ec-0524bd3077c0.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210251831-d76514ff-27c3-4b47-bbd0-976379152674.mp4

![988](https://user-images.githubusercontent.com/117115792/209517414-a68d633f-e111-49a9-a1d5-2114ae74ef75.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210251861-6afb3fd5-9fea-4d2c-8b70-18c1d62efdcb.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210251877-b77e2f66-2d87-42ea-a4c6-9567bc978929.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210251898-4d3fc8f7-1c5b-42cd-ab67-15c9f9c69df8.mp4

![1000](https://user-images.githubusercontent.com/117115792/209517422-11ee15d6-15de-407b-a8c5-db4d6aed2e7f.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210251909-f4ff3544-01a9-491b-ba0c-aaa9bb94c579.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210251916-06430c21-51fd-4962-9723-10e24a2c4ee5.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210251935-9f74a6e9-69f6-4e81-9fc0-96d8c6bde4f9.mp4

![1007](https://user-images.githubusercontent.com/117115792/209517429-2bd68315-37d0-4cfd-bc89-5bfe2cef5dcd.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210251939-a0317644-742c-4107-b1fe-bcda66bc19eb.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210251946-cb1a9a0b-f7b9-43d5-82df-ada33779779d.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210251955-d2b67cc9-8fe1-4b30-bc12-66633eb278be.mp4

![1008](https://user-images.githubusercontent.com/117115792/209517440-ac2d9068-4b20-4ddd-adb6-735edd9af2cf.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210251968-68070ed8-adc0-4014-8898-28548acf929d.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210251982-20538f03-71cb-4d11-9407-ff9b6411e64c.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210251988-3dce021a-1df3-4906-976f-cbadadb9dcfd.mp4

![1009](https://user-images.githubusercontent.com/117115792/209517444-1562c27b-24cc-436a-b8b7-88cbb5f33b70.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210252006-9a59e774-673d-4196-a0c3-790050a9b642.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210252014-472190c7-5be6-43df-9526-beae61325207.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210252038-13661baf-0a40-4fe4-9761-3b326eb44a19.mp4

![1027](https://user-images.githubusercontent.com/117115792/209517448-7494655d-e45b-43cc-85f3-21c1d66ecae1.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210252052-41528775-a298-49ab-a88a-e1fb408a28cb.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210252071-6cd46c1b-14c2-4ae3-b32f-0b2f9b357f74.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210252082-d815b350-6f7b-4a85-9bff-befcd1473c05.mp4

![1034](https://user-images.githubusercontent.com/117115792/209517457-6de66000-d116-4a0b-b9e8-1ffba4ce5483.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210252119-8effb132-8ddd-43fe-85a0-5674bfcc84e1.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210252128-467c9ca1-4c33-4631-8a1f-f88429e5677f.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210252144-8a2dd1a2-179a-4200-bed9-2cd11a72d6e0.mp4

![1265](https://user-images.githubusercontent.com/117115792/209517470-31c87daf-70ce-494f-b0fc-ff6eb90e1782.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210252157-32e77f48-07f6-4858-b7c1-6a9a79c69ea7.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210252169-aa3ae1c6-6aad-405b-8f3a-104b2bcc7d92.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210252175-388fdd2a-e21b-406d-84c1-999a6af4040a.mp4

![1284](https://user-images.githubusercontent.com/117115792/209517474-e37ac407-ffd6-49a5-b942-83ba7d113ab5.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/210252182-83849fc6-54ad-44e8-b9fd-8120fefde265.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/210252196-927044ec-2d26-4224-a154-d1a20cb2a111.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/210252214-10d3a07e-7192-4544-a5ca-d004995563a6.mp4

![7](https://user-images.githubusercontent.com/117115792/209510044-2fde7e8c-9151-4b79-bc05-202a3ee8b9c2.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/209516240-995b3842-820f-479c-b901-1d30fe112c57.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/209516255-ec9a791e-9e28-4465-92ce-c40db6671619.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/209516266-f7b7a368-df33-41c2-a102-501876b1d7c7.mp4


![32](https://user-images.githubusercontent.com/117115792/209510142-75efddbd-483f-4df0-8ded-1e12b39b13a7.png)

Clean Signal:

https://user-images.githubusercontent.com/117115792/209516633-4586be01-98d1-47ca-9864-223eb130da69.mp4

Corrupted Signal:

https://user-images.githubusercontent.com/117115792/209516644-fb511f32-4c2a-40a1-9ff1-55104b9cb48d.mp4

Restored Signal:

https://user-images.githubusercontent.com/117115792/209516657-6239ef67-370b-4853-a63b-61a2d19b5ef0.mp4

## Citation
If you find this project useful, we would be grateful if you cite this paper：

```http

```
