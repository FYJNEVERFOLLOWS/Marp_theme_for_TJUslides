---
marp: true
theme: gaia
paginate: true
footer: 'Reported by FuYanjie 2021-09-24'
style: | 
  section footer{color:black;font-size: 20px;} 


math: katex
---
<style scoped>
section h1 {text-align: center;font-size: 80px;color:black;}
section {
  background-image:url('/Users/fuyanjie/Desktop/PG/TJU elements.png/fm.png');
  background-size:cover
}
footer{color:black;font-size: 20px;} 
</style>
<!-- _class: lead gaia -->

# DNN for Multiple Speaker Detection and Localization
## FuYanjie 2021.9.24

---
<style >

section{
  background-image:url('/Users/fuyanjie/Desktop/PG/TJU elements.png/bg.png');
  background-size:cover;
  position: absolute;
  }
section h1 {font-size:40px;color:black;margin-top:px;}
section h2 {font-size:30px;color:black;margin-top:px;}
section p {font-size: 25px;color:black;}
section table {text-align: center;font-size: 32px;color:black;}
section a {font-size: 25px;color:black;}
li {font-size: 30px;text-align: left;}

img {
    margin-left: auto; 
    margin-right:auto; 
    display:block;
    margin:0 auto;
    width:25cm;
    }
</style>

<style scoped>
section h1 {font-size:60px;color:black;margin-top:px;}
li {font-size: 50px;text-align: left;}
</style>

# Overview
1. Introduction
2. Proposed Method
3. Experiment
4. Conclusion

[Author's homepage: https://idiap.ch/~whe/](https://idiap.ch/~whe/)

[Paper's link: https://arxiv.org/pdf/1711.11565.pdf](https://arxiv.org/pdf/1711.11565.pdf)

---
# Introduction
## Challenges
1. Noisy environments;
2. Multiple simultaneous speakers;
3. Short and low-energy utterances;
4. Obstacles such as robot body blocking sound direct path. 

---
# Introduction
## Motivation
###### Compared to Conventional Methods:

Conventional signal processing algorithms are derived with assumptions, many of which do not hold well under the above-mentioned conditions.

NNs can learn the mapping from the localizaition cues to the direction-of-arrival without making strong assumptions.

---
# Introduction
## Motivation
###### Compared to Existing NN-based SSL Methods:
1. Do not address the problem of multiple sound sources;
2. Cannot detect and localize multiple voices in real multi-party HRI scenarios simultaneously.
3. Formulate the problem as the classification of an audio input into one "class" label associated with a location, and optimizing the posterior probability of such labels. Such posterior probability encoding cannot be easily extended to multiple sound source situations.

---
# Introduction
## Contributions
![w:32cm](https://tva1.sinaimg.cn/large/008i3skNly1gulrsxx4pwj60vd07tdhi02.jpg)

Our methods can cope with short input, overlapping speech, an unknown number of sources and strong ego-noise.

---
# Proposed Method
## A. Input Features
time frame: 170ms&emsp;&emsp;number of sources: $N$ &emsp;&emsp;number of microphones: $M$

STFT of input signal: $X_i(\omega),i=1,...,M$, the mic index: $i$, the discrete freq: $\omega$.

Two types of features based on GCC-PHAT at frame level:

* GCC-PHAT coefficients: 

The GCC-PHAT between channel $i$ And $j$ Is formulated as:
$$
g_{i j}(\tau)=\sum_{\omega} \mathcal{R}\left(\frac{X_{i}(\omega) X_{j}(\omega)^{*}}{\left|X_{i}(\omega) X_{j}(\omega)^{*}\right|} e^{j \omega \tau}\right)
$$

---
# Proposed Method

$\tau$$(\in[-25,25])$ is the discrete delay (use the center 51 delays), (·)* denotes the complex conjugation, $\mathcal{R}(·)$ denotes the real part of a complex number.

* GCC-PHAT on Mel-scale filter bank: 

The GCC-PHAT is not optimal for TDOA estimation of multiple source signals since it equally sums over all freq bins disregarding the "sparsity" of speech signals in the TF domain. Thus propose to use GCC-PHAT on Mel-scale filter bank (GCCFB).
$$
g_{i j}(f, \tau)=\frac{\sum_{\omega \in \Omega_{f}} \mathcal{R}\left(H_{f}(\omega) \frac{X_{i}(\omega) X_{j}(\omega)^{*}}{\left|X_{i}(\omega) X_{j}(\omega)^{*}\right|} e^{j \omega \tau}\right)}{\sum_{\omega \in \Omega_{f}} H_{f}(\omega)}
$$
$f$ is the filter index, $H_f$ is the transfer function of the f-th Mel-scaled triangular filter.
40 Mel-scale filters covering the frequencies from 100 to 8000 Hz.

---
# Proposed Method
![w:18cm](https://tva1.sinaimg.cn/large/008i3skNly1guks65tdq6j60ns0ir40g02.jpg)

---
# Proposed Method
## B. Likelihood-based Output Coding
&nbsp; 
**Encoding:** the output (the likelihood of a sound source being in each direction) is encoded into a vector {$o_i$} of 360 values (分别对应 $\theta_i$ ). The values are defined as the maximum of Gaussian-like functions centered around the true DOAs:
$$
o_{i}= \begin{cases}\max _{j=1}^{N}\left\{e^{-d\left(\theta_{i}, \theta_{j}^{(s)}\right)^{2} / \sigma^{2}}\right\} & \text { if } N>0 \\ 0 & \text { otherwise }\end{cases}
$$
其中 $\theta_j^{(s)}$ 是第 j 个声源DOA的真实值，$\sigma$ 是高斯分布的标准差（尺度参数），$d(·,·)$ 表示 angular distance

---
# Proposed Method
## B. Likelihood-based Output Coding
![bg right:45% w:20cm](https://tva1.sinaimg.cn/large/008i3skNly1gukvyfbhf8j60mo07hq3j02.jpg)
Posterior probability coding is constrained as a probability distribution (the output layer is normalized by a softmax function). It can be all zero when there is no sound source, or contains N peaks when there are N sources.

---
# Proposed Method
## B. Likelihood-based Output Coding
&nbsp; 
**Decoding:** During the test phase, we decode the output by finding the peaks that are above a given threshold $\xi$ :
$$
\text { Prediction }=\left\{\theta_{i}: o_{i}>\xi \quad \text { and } \quad o_{i}=\max _{d\left(\theta_{j}, \theta_{i}\right)<\sigma_{n}} o_{j}\right\}
$$
with $\sigma_n$ being the neighborhood distance. We choose $\sigma = \sigma_n = 8°$ for the experiments.

---
# Proposed Method
## C. 3 different Neural Network Architectures
**MLP-GCC (Multilayer perceptron with GCC-PHAT)**
![bg right:45% w:8cm](https://i.postimg.cc/pVkvZPS9/WX20210920-145332.png)

Three hidden fully-connected layers with ReLU activation function and Batch Normalization

The last fully-connected layer with sigmoid activation function

---
# Proposed Method
## C. 3 different Neural Network Architectures
**CNN-GCCFB (Convolutional neural network with GCCFB)**
![bg right:46% w:11cm](https://i.postimg.cc/rmPJV0m0/WX20210920-145623.png)

FC NNs are not suitable for high-dimensional input features (such as GCCFB) (introduces a large amount of parameters; prone to overfitting).

Four convolutional layers (with ReLU activation and BN) and a FC layer at the output (with sigmoid activation)

---
# Proposed Method
## C. 3 different Neural Network Architectures
**TSNN-GCCFB (Two-stage neural network with GCCFB)**
* do analysis or implicit DOA estimation in each freq band before such info is aggregated into a broadband prediction
* Features with the same delay on different microphone pairs do not correspond to each other locally. Instead, feature extraction or filters should take the whole delay axis into account.

Training scheme: First, we train the Subnet 1 in the first stage using the DOA likelihood as the desired latent feature. During the second step, both stages are trained in an end-to-end manner.

---
# Proposed Method
## C. 3 different Neural Network Architectures
![h:13cm](https://tva1.sinaimg.cn/large/008i3skNly1gul55zr3hxj60nt0h9gny02.jpg)

---
# Experiment
## C. Network Training
Adam optimizer

MSE loss

Mini-batch size: 256

10 epochs for MLP-GCC & CNN-GCCFB

4 epochs for the first stage of TSNN-GCCFB and 10 epochs for the end-to-end training

---
# Experiment
## A. Datasets
sr: 48kHz
4 mics, rectangle of 5.8 X 6.9 cm

Loudspeaker Recordings

├── lsp_train_106 (a large conference room)
├── lsp_train_301 (a small conference room)
├── lsp_test_106
├── lsp_test_library (a small room with shelves)

├── lsp_<*>
│   ├── audio
│   ├── gt_file
│   └── gt_frame

---
# Experiment
## A. Datasets
**Audio Files:** "RECORD_ID.wav"

**File-Level Ground Truth:**
* the recording ID
* the start & end time of the recording in the original ROS bag file
* list of source labels, each source label is a tuple of:
  * 3D source location
  * source audio file (a segment from the AMI corpus)
  * the start & end time of the source in the recording
  * relative volume of the source

---
# Experiment
## A. Datasets
**Frame-Level Ground Truth:**
* the frame ID, which start from 0. The frame of ID t contains samples between [t * HOP_SIZE, t * HOP_SIZE + WIN_SIZE).
* list of active sources (can be empty list if there is no active source). Each active source contains:
  * 3D source location
  * source type, which is always 1 (speech source).
  * speaker ID

---
# Experiment
## A. Datasets
Human Talker Recordings
├── human
│   ├── audio
│   ├── audio_gt
│   ├── gt_frame
│   └── video_gt

**Audio Ground Truth:**

The audio_gt directory includes the voice activity ground truth.

**Video Data and Ground Truth:**

The video_gt directory includes the video data and source location ground truth.

---
# Experiment
## B. Evaluation Protocol
We evaluate multiple SSL methods at frame level under two different conditions: the number of sources is known or unknown.

Known: 
We select the N highest peaks of the output as the predicted DOAs and match them with ground truth DOAs one by one, and we compute the mean absolute error (MAE). Evaluate by ACC of predictions.
By saying a prediction is correct, we mean the error of the prediction is less than a given admissible error $E_a$.

Unknown: 
detection - given ground truth sources, compute recall (the percentage of correct detection out of all ground truth sources)
localization - compute precision (the percentage of correct predictions among all predictions)

---
# Experiment
## D. Baseline Methods
Spatial spectrum-based methods:

SRP-PHAT

SRP-NONLIN: SRP-PHAT with a non-linear modification of the score

MVDR-SNR: minimum variance distortionless response beam forming with signal-to-noise ratio as score

SEVD-MUSIC: multiple signal classification, assuming spatially white noise and one signal in each bin

GEVD-MUSIC: MUSIC with generalized eigenvector decomposition, assuming noise is pre-measured and one signal in each TF bin

---
# Experiment
## E. Results
![w:30cm](https://tva1.sinaimg.cn/large/008i3skNly1gulp4sdms0j615g0cq77p02.jpg)

---
# Experiment
## E. Results
![h:12cm](https://tva1.sinaimg.cn/large/008i3skNly1gulpbp9xe0j60tw0mon2202.jpg)

---
# Conclusion
**Limitation:**

* The current study is potentially limited by the training data samples, which are not likely to cover all possible combinations of source positions, since the number of combinations grows exponentially with the number of sources. 

* Will investigate the incorporation of temporal context.

---
<style scoped>
section h1 {text-align: center;font-size: 100px;color:black;}
section {
  background-image:url('./fm.png');
  background-size:cover
}
footer{color:black;font-size: 20px;} 
</style>
<!-- _class: lead gaia -->

# Thanks for your time!