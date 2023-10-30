# AdvNIR

Welcome! This is the official implementation for our paper:

💣 📷 [ACM MM 2023] Physics-Based Adversarial Attack on Near-Infrared Human Detector for Nighttime Surveillance Camera Systems

by [Muyao Niu](https://myniuuu.github.io), Zhuoxiao Li, Yifan Zhan, Huy H. Nguyen, Isao Echizen, and Yinqiang Zheng.

> Many surveillance cameras switch between daytime and nighttime modes based on illuminance levels. During the day, the camera records ordinary RGB images through an enabled IR-cut filter. At night, the filter is disabled to capture near-infrared (NIR) light emitted from NIR LEDs typically mounted around the lens. While the vulnerabilities of RGB-based AI algorithms have been widely reported, those of NIR-based AI have rarely been investigated. In this paper, we identify fundamental vulnerabilities in NIR-based image understanding caused by color and texture loss due to the intrinsic characteristics of clothes' reflectance and cameras' spectral sensitivity in the NIR range. We further show that the nearly co-located configuration of illuminants and cameras in existing surveillance systems facilitates concealing and fully passive attacks in the physical world. Specifically, we demonstrate how retro-reflective and insulation plastic tapes can manipulate the intensity distribution of NIR images. We showcase an attack on the YOLO-based human detector using binary patterns designed in the digital space (via black-box query and searching) and then physically realized using tapes pasted onto clothes. Our attack highlights significant reliability concerns about nighttime surveillance systems, which are intended to enhance security.

## Envionment Setup
```
conda create -n advnir python==3.8
conda activate advnir
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install blenderproc
pip install tqdm
```
## Data Prepraration

### Download prerequisite data

Download prerequisite data from [here](https://drive.google.com/file/d/1qsOxa4E6SMBs_2apNzbPXOB8b2BEA3TI/view?usp=drive_link), then unzip them to `./`.

## Synthesize Meshes and 2D Segment Maps

The following commands discribe how to synthesize 3D meshes and render 2D segment maps. If you do not want to run these codes, you can simply download our pre-synthesized 3D meshes and 2D segment maps from [here](https://drive.google.com/file/d/1fHCbpk0e-pz2gVw_YkIGNAMkvjhp6rHN/view?usp=drive_link), and unzip the file to `./`. Then you can ignore the following commands of this section.

### Synthesize human body meshes

```
blenderproc run generate_mesh.py
```

The generated human body meshes will be saved to `./generated_meshes` by default.

### Split human body meshes

```
python split_mesh.py
```

The splited human body meshes will be saved to `./splited_meshes` by default.

### Render 2D Segment Maps

```
blenderproc run render_2d_map_train.py
blenderproc run render_2d_map_test.py
```

The rendered 2D segment maps will be saved to `./rendered_2d_maps` by default. each `.npy` file is the segment map which will be used by our algorithm, and the corresponding `.png` file is just for visualization.


## Design the Adversarial Patterns

Run the following commands to search for the adversarial patterns. As has been mentioned in our paper, we have two options:

1. 1black: forcing head to black:

    ```
    python search_1black.py
    ```

2. 5black: forcing head, hands, and foots’ part to black.

    ```
    python search_5black.py
    ```


## Citation
If you find this repo useful, please consider citing:

```
@inproceedings{10.1145/3581783.3612082,
    author = {Niu, Muyao and Li, Zhuoxiao and Zhan, Yifan and Nguyen, Huy H. and Echizen, Isao and Zheng, Yinqiang},
    title = {Physics-Based Adversarial Attack on Near-Infrared Human Detector for Nighttime Surveillance Camera Systems},
    year = {2023},
    booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
    pages = {8799–8807},
}
```