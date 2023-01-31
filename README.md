# Few shot 3D reconstruction using Score Distillation Sampling

An implementation using [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion) that applies the Text-to-3D model [DreamFusion](https://dreamfusion3d.github.io/) to the task of Few-shot / single-view 3D object reconstruction. 

## Examples

### Few-shot 3D object reconstruction

Input RGB views (only 3 sparse views, omitting the right side of the car):
| 1 | 2 | 3 |
| - | - | - |
| ![car_rgba_00002](https://user-images.githubusercontent.com/21123989/215728220-427cee0e-3623-4877-9d2f-1bfd85c837c7.png) | ![car_rgba_00000](https://user-images.githubusercontent.com/21123989/215728247-ecc4c504-8f4b-450c-9d27-25fee0d9e34a.png) | ![car_rgba_00001](https://user-images.githubusercontent.com/21123989/215728277-3bd19476-abaf-4711-977e-0bc020c054fb.png) |

L2 reconstruction loss on the input views + Score Distillation Sampling with the guiding prompt "a yellow sports car":

[car_reconstruction_1.webm](https://user-images.githubusercontent.com/21123989/212968476-8c7503d0-fb21-45af-892c-79dd9d2b1eaf.webm)

### Text-to-3D

Score Distillation Sampling with a guiding prompt:

| 1 | 2 | 3 |
| - | - | - |
| ![croppedsample_3_angle_oh](https://user-images.githubusercontent.com/21123989/215727709-6aea580d-5c7d-4834-81e2-e47041a9ed47.png) | ![croppedsample_3](https://user-images.githubusercontent.com/21123989/215727737-b5ebcad1-845c-4e74-a31d-188568670f58.png) | ![croppedsample_3_normal](https://user-images.githubusercontent.com/21123989/215727760-200d4607-b2aa-4685-b324-fc522122435b.png) |
| ![croppedsample_2](https://user-images.githubusercontent.com/21123989/215727843-38c2b532-6891-4bcc-b1aa-955ae5627433.png) | ![croppedsample_2_normal](https://user-images.githubusercontent.com/21123989/215727876-29064336-0eec-4dbe-8092-d81714291efc.png) | ![croppedsample_2_albedo](https://user-images.githubusercontent.com/21123989/215727912-edba026b-c951-41bc-b865-415cc1a6e313.png) |
| ![croppedsample_1](https://user-images.githubusercontent.com/21123989/215727601-d7653f4d-28b1-4c50-86fe-9023cc6efcd3.png) | ![croppedsample_1_normal](https://user-images.githubusercontent.com/21123989/215727985-7c4531bc-1c9f-4aef-828f-e2c1790f32b0.png) | ![croppedsample_4](https://user-images.githubusercontent.com/21123989/215728029-4870d271-312b-4645-b70c-f32ee115f28a.png) |

## Installation

Follow the instrutions [here](https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/INSTALL.md) to install [NVIDIA Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp).

Run ```setup.sh``` to install the prerequisites.


## Usage

### Few-shot 3D object reconstruction

```python main.py --config configs/diffusion_nerf.yaml --dataset-path /path/to/car/dataset --prompt "a car"```

### Text-to-3D

```python main.py --config configs/diffusion_nerf.yaml --prompt "a DSLR photo of a blue car"```



