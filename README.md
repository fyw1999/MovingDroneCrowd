# Video Individual Counting for Moving Drones (ICCV 2025)
## Introduction
This is the official PyTorch implementation of paper: [Video Individual Counting for Moving Drones](https://arxiv.org/abs/2503.10701), which introduce a video-level individual counting dataset captured by fast-moving drones in various crowded scenes and propose a shared density map-guided VIC method
that bypasses the challenging localization step and instead adopts a more learnable manner by first learning shared pedestrian density maps between consecutive frames.

![pipeline](figures/pipeline.jpg)

# Catalog
✅ MovingDroneCrowd

✅ Training and Testing Code

✅ Pretrained models for MovingDroneCrowd

# MovingDroneCrowd
To promote practical crowd counting, we introduce MovingDroneCrowd — a video-level dataset specifically designed for dense pedestrian scenes captured by moving drones under complex conditions. **Notably, our dataset provides precise bounding box and ID labels for each person across frames, making it suitable for multiple pedestrian tracking from drone perspective in complex scenarios.**

![dataset_example](figures/dataset_example.jpg)

MovingDroneCrowd are available at the [Google Drive](https://drive.google.com/file/d/1VufYjfFBFA96UCHK6XJYhgQBkRKokQte/view?usp=drive_link), 




# Citation
If you find this project is useful for your research, please cite:

```bibtex
@article{MVC,
  title={Video Individual Counting for Moving Drones},
  author={Fan, Yaowu and Wan, Jia and Han, Tao and Chan, Antoni B and Ma, Andy J},
  booktitle={ICCV},
  year={2025}
}

# Acknowledgement

The released PyTorch training script borrows some codes from the [DRNet](https://github.com/taohan10200/DRNet). If you think this repo is helpful for your research, please consider cite them.