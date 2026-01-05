<h1 align='center'>TexAvatars: Hybrid Texel–3D Representations for Stable Rigging of Photorealistic Gaussian Head Avatars</h1>


<div align="center">
    <a href="https://leejesse.github.io/" target="_blank">Jaeseong Lee</a><sup>1*</sup>&emsp;
    <a href="https://justin4ai.github.io/" target="_blank">Junyeong Ahn</a><sup>2*</sup>&emsp;
    <a href="https://keh0t0.github.io/" target="_blank">Taewoong Kang</a><sup>1</sup>&emsp;
    <a href="https://sites.google.com/site/jaegulchoo/" target="_blank">Jaegul Choo</a><sup>1</sup>
</div>
<div align="center">
    <sup>1</sup>KAIST AI&emsp;
    <sup>2</sup>Hanyang University
</div>
<div align="center">
    <sup>*</sup> denotes equal contribution.
</div>

<br>
<div align='center'>
    <a href='https://summertight.github.io/TexAvatars/'><img src='https://img.shields.io/badge/Project-HomePage-3A4F7A?style=flat'></a>
    <a href='https://arxiv.org/abs/2512.21099'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

</div>



## 📸 Teaser



https://github.com/user-attachments/assets/fb5e97f2-a097-4ff3-9f05-56bbdb50da5a


<br>

## ⚙️ Installation
- System requirement: Ubuntu 20.04/Ubuntu 22.04, CUDA 11.8
- Tested GPUs: RTX2080Ti, 3090Ti

```bash
git clone https://github.com/summertight/TexAvatars_repo.git --recursive
cd TexAvatars_repo

conda create -n texavatars python=3.10
conda activate texavatars
```


(Optional) If your environment doesn't match CUDA 11.8:
```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit ninja
```


Install packages:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

<br>

## 💾 Download
### 1. Dataset
Our code uses ```NeRSemble``` multi-view dataset. Please refer to [NeRSemble](https://github.com/tobias-kirschstein/nersemble-data) github and submit your agreement([link](forms.gle/rYRoGNh2ed51TDWX9)) to download. 

A detailed instruction of pre-processing and FLAME pose tracking is already provided by [VHAP](https://github.com/ShenhanQian/VHAP/blob/main/doc/nersemble.md).

We provide the fully-processed dataset of ID ```074``` and ```175``` in [Google Drive](https://drive.google.com/drive/folders/1OmBkZgDazsBdAXi2dToRI_1t_snu7q_7?usp=sharing). Download them and place under PATH/TO/DATASET then unzip each. An expected folder structure is as follows:

```bash
PATH/TO/DATASET/
├── 074_EMO-1_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_...
│   ├── images/
│   ├── fg_masks/
│   ├── flame_param/
│   ├── emo_emb/                  # EMOPortraits embeddings
│   ├── canonical_flame_param.npz
│   ├── transforms.json
│   ├── transforms_train.json
│   ├── transforms_val.json
│   └── transforms_test.json
│   ...
├── 074_EXP-1_...
│   └── (same structure)
│   ...
└── UNION10_074_EMO1234EXP234589_v16_DS2-...
    ├── canonical_flame_param.npz
    ├── sequences_test.txt
    ├── sequences_trainval.txt
    ├── transforms_train.json
    ├── transforms_val.json
    └── transforms_test.json
```

### 2. FLAME Model
Our code relies on FLAME 2023. Please download [FLAME assets](https://flame.is.tue.mpg.de/download.php) and place them in following paths:

- FLAME2023 (with jaw) -> `flame_model/assets/flame/flame2023.pkl`
- FLAME_masks -> `flame_model/assets/flame/FLAME_masks.pkl`



### 3. EMOPortraits Embeddings
For each frame, you need to obtain embeddings of [EMOPortraits](https://github.com/neeek2303/EMOPortraits). Please follow the instruction of [TexAvatars-EMO](https://github.com/justin4ai/TexAvatars-EMO).

We provide extracted embeddings for ID ```074``` and ```175```, which are already within the provided processed dataset.


### 4. Pretrained Weights

We provide the checkpoint for ID ```074```. Download via [Google Drive](https://drive.google.com/drive/folders/1OmBkZgDazsBdAXi2dToRI_1t_snu7q_7?usp=sharing) and place the folder under ```./output/```. ```./``` indicates your current cloned TexAvatars path.
<br>

## 🚀 Render
### Self-Reenactment
Edit ```self_reenact.sh``` with proper dataset and saving path. Then run
```bash
bash self_reenact.sh $GPU_ID $SUBJ_ID
```

. For example,

```bash
bash self_reenact.sh 0 074
```

### Cross-Reenactment

Edit ```cross_reenact.sh``` with proper dataset and saving path. Then run
```bash
bash cross_reenact.sh $GPU_ID $SUBJ_ID $DRIVER_ID
```

. For example,

```bash
bash cross_reenact.sh 0 074 175
```


## 🗝️️ Training

Edit ```train.sh``` with proper dataset and saving path. Then run
```bash
bash train.sh $GPU_ID $SUBJ_ID
```

. For example,

```bash
bash train.sh 0 074
```


<br>
