# Latent Spectral Models (ICML 2023)

Solving High-Dimensional PDEs with Latent Spectral Models [[paper]](https://arxiv.org/abs/2301.12664)

To tackle both the approximation and computation complexities in PDE-governed tasks. We propose the Latent Spectral Models (LSM) with the following features:

- Free from unwieldy coordinate space, LSM **solves PDEs in the latent space**.
- LSM holds the **universal approximation capacity** under theoretical guarantees.
- LSM achieves **11.5% error reduction over previous SOTA in both solid and fuild physics** and performs favorable **efficiency and transferability**.

<p align="center">
<img src=".\fig\model.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of LSM.
</p>

## LSM vs. Previous Methods

Different from previous methods that learn a single operator directly, inspired by classical spectral methods in numerical analysis, LSM composes complex mappings into multiple basis operators. Along with the latent space projection, LSM presents favorable approximation and convergence properties.

<p align="center">
<img src=".\fig\compare.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 2.</b> Comparison in approximating complex mappings.
</p>

## Get Started

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the datasets from the following links.


| Dataset       | Task                                    | Geometry        | Link                                                         |
| ------------- | --------------------------------------- | --------------- | ------------------------------------------------------------ |
| Elasticity-P  | Estimate material inner stress          | Point Cloud     | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Elasticity-G  | Estimate material inner stress          | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | Estimate material deformation over time | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Navier-Stokes | Predict future fluid velocity           | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| Darcy         | Estimate fluid pressure through medium  | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| AirFoil       | Estimate airﬂow velocity around airfoil | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | Estimate fluid velocity in a pipe       | Structured Mesh | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

2. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/elas_lsm.sh # for Elasticity-P
bash scripts/elsa_interp_lsm.sh # for Elasticity-G
bash scripts/plas_lsm.sh # for Plasticity
bash scripts/ns_lsm.sh # for Navier-Stokes
bash scripts/darcy_lsm.sh # for Darcy
bash scripts/airfoil_lsm.sh # for Airfoil
bash scripts/pipe_lsm.sh # for Pipe
```

## Results

We extensively experiment on seven benchmarks and compare LSM with 13 baselines. LSM achieves the consistent state-of-the-art in both solid and fluid physics (11.5% averaged error reduction).

<p align="center">
<img src=".\fig\main_results.png" height = "350" alt="" align=center />
<br><br>
<b>Table 1.</b> Model perfromance on seven benchmarks. MSE is recorded.
</p>

## Showcases

<p align="center">
<img src=".\fig\showcases.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 3.</b> Showcases. LSM can capture the shock wave around the airfoil precisely.
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2023LSM,
  title={Solving High-Dimensional PDEs with Latent Spectral Models},
  author={Haixu Wu and Tengge Hu and Huakun Luo and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Contact

If you have any questions or want to use the code, please contact [whx20@mails.tsinghua.edu.cn](mailto:whx20@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO
