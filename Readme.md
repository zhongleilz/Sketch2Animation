# Sketch2Anim: Towards Transferring Sketch Storyboards into 3D Animation
### [Project Page](https://zhongleilz.github.io/Sketch2Anim/) | [Paper](https://arxiv.org/pdf/2504.19189)

> Sketch2Anim: Towards Transferring Sketch Storyboards into 3D Animation

> [Lei Zhong](https://zhongleilz.github.io/), [Chuan Guo](https://ericguo5513.github.io/), [Yiming Xie](https://ymingxie.github.io),[Jiawei Wang](https://jiawei22.github.io/), [Changjian Li](https://enigma-li.github.io/)

![teaser](assets/teaser.gif)

## Citation
If you find our code or paper helpful, please consider starring our repository and citing:
```bibtex
@Article{Zhong:2025:Sketch2Anim, 
    Title = {Sketch2Anim: Towards Transferring Sketch Storyboards into 3D
    Animation}, 
    Author = {Lei Zhong, Chuan Guo, Yiming Xie, Jiawei Wang and Changjian Li}, 
    Journal = {ACM Transaction on Graphics (TOG)},
    volume={44},
    number={4},
    pages={1--15},
    Year = {2025}, 
    Publisher = {ACM New York, NY, USA} 
} 

```

## TODO List
- [] Code for Inference and Pretrained model.
- [] Evaluation code and metrics.
- [] Code for training.
- [] Blender Plugin.


<!-- ## PRETRAINED_WEIGHTS
Available on [Google Drive](https://drive.google.com/drive/folders/12m_v_vybVeAQFkH9bP8wmJIxJhGoIJL1?usp=sharing).

## Getting started
This code requires:

* Python 3.9
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate omnicontrol
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```


### 2. Get data

#### Full data (text + motion capture)

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**100STYLE** - Download the dataset from Google Drive, then copy the files in texts, new_joints, and new_joint_vecs into their corresponding directories within ./dataset/HumanML3D. We use indices larger than 030000 to represent data from the 100STYLE dataset.

### 3. Download the pretrained models

1. Download the model(s) you wish to use, then unzip and place them in `./save/`. 
2. Download the pretrained model from [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) and then copy it to `./save/`. 


## Motion Synthesis
Please add the content text to ./demo/test.txt and the style motion to ./test_motion, then run:
```shell
bash demo.sh
```

Tips:
1. For some motion styles, the default parameter settings may not achieve the desired results. You can modify the `guidance_scale_style` in `config_cmld_humanml3d.yaml` to achieve a better balance between content preservation and style reflection.
2. Make sure to set `is_test: True`.

## Train your own SMooDi
You can train your own model via
```shell
bash train.sh
```

Tips:
1. In `config_cmld_humanml3d.yaml`, set `is_recon: True` means that cycle loss will not be used during training. 
2. In `config_cmld_humanml3d.yaml`, set `guidance_mode: v0` for training.
3. In fact, the improvement in performance from cycle loss is quite limited. If you want to quickly train a model, you can set `is_recon: True`. With this setting, it will take nearly 50 minutes to train 50 epochs on an A5000 GPU and achieve performance nearly equivalent to the second row in Table 3 of our paper. 


## Evaluate
You can evaluate model via
```shell
bash test.sh
```


Tips:
1. In `config_cmld_humanml3d.yaml`, set `guidance_mode: v2 or v4` for evaluation.
2. Make sure to set `is_test: True` during evaluation.
3. In `config_cmld_humanml3d.yaml`, set `is_guidance: True` means that classifier-based style guidance will be used during evaluation. If `is_guidance: False`, evaluation will take nearly 50 minutes, whereas it will take 4 hours if `is_guidance: True` on an A5000 GPU. -->

## Acknowledgments

Our code is heavily based on [MLD](https://github.com/ChenFengYe/motion-latent-diffusion).  
The motion visualization is based on [MLD](https://github.com/ChenFengYe/motion-latent-diffusion) and [TMOS](https://github.com/Mathux/TEMOS). 
We also thank the following works:
[guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [OmniControl](https://github.com/neu-vi/OmniControl).

## License
This code is distributed under an [MIT LICENSE](LICENSE).  

Note that our code depends on several other libraries, including SMPL, SMPL-X, and PyTorch3D, and utilizes the HumanML3D datasets. Each of these has its own respective license that must also be adhered to.
