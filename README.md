# Few shot 3D reconstruction using Score Distillation Sampling

An implementation using [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion) that applies the Text-to-3D model [DreamFusion](https://dreamfusion3d.github.io/) to the task of Few-shot / single-view 3D object reconstruction. 

## Installation

Follow the instrutions [here](https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/INSTALL.md) to install [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp).

Run ```setup.sh``` to install the prerequisites.


## Running

### Text-to-3D

```python main.py --config configs/diffusion_nerf.yaml --prompt "a DSLR photo of a blue car"```

### Few-shot 3D object reconstruction

```python main.py --config configs/diffusion_nerf.yaml --dataset-path /path/to/car_dataset --prompt "a car"```



