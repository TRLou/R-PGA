# R-PGA: Robust Physical Adversarial Camouflage Generation via Relightable 3D Gaussian Splatting
This repository contains the official implementation of R-PGA. The full paper link will be available soon.


**Abstract**—Physical adversarial camouflage poses a severe security threat to autonomous driving systems by mapping adversarial
textures onto 3D objects. Nevertheless, current methods remain brittle in complex dynamic scenarios, failing to generalize across
diverse geometric (e.g., viewing configurations) and radiometric (e.g., dynamic illumination, atmospheric scattering) variations. We
attribute this deficiency to two fundamental limitations in simulation and optimization. First, the reliance on coarse, oversimplified
simulations (e.g., via CARLA) induces a significant domain gap, confining optimization to a biased feature space. Second, standard
strategies targeting average performance result in a rugged loss landscape, leaving the camouflage vulnerable to configuration shifts.
To bridge these gaps, we propose the Relightable Physical 3D Gaussian Splatting (3DGS) based Attack framework (R-PGA).
Technically, to address the simulation fidelity issue, we leverage 3DGS to ensure photo-realistic reconstruction and augment it with
physically disentangled attributes to decouple intrinsic material from lighting. Furthermore, we design a hybrid rendering pipeline that
leverages precise Relightable 3DGS for foreground rendering, while employing a pre-trained image translation model to synthesize
plausible relighted backgrounds that align with the relighted foreground. To address the optimization robustness issue, we propose the
Hard Physical Configuration Mining (HPCM) module, designed to actively mine worst-case physical configurations and suppress their
corresponding loss peaks. This strategy not only diminishes the overall loss magnitude but also effectively flattens the rugged loss
landscape, ensuring consistent adversarial effectiveness and robustness across varying physical configurations. Extensive
experiments confirm R-PGA’s state-of-the-art performance and superior robustness in both digital and physical domains, where it
outperforms the best-competing baselines by further reducing the average AP@0.5 by 6.56% and 6.12%, respectively

## Acknowledgments
This project is built upon the following open-source repositories: [LBM](https://github.com/gojasper/LBM.git) and [GIR](https://github.com/guduxiaolang/GIR.git) 

## Get Started
**Step 1. Environment Setup**

Please follow the installation instructions in the [LBM](https://github.com/gojasper/LBM.git) and [GIR](https://github.com/guduxiaolang/GIR.git) repositories to configure the core requirements.
Once the base environment is ready, install the remaining Python packages:

`pip install -r requirements.txt`

**Step 2. Data Preparation & GIR Training**

Before generating adversarial camouflage, you need to reconstruct the target object using GIR.

Data Preparation: Organize your multi-view images and camera parameters according to the GIR format.

Controlling Complexity: To generate textures with different levels of granularity, you must control the number of Gaussian ellipsoids.

Note: Adjust the densification/splitting conditions during GIR training to limit or expand the Gaussian count for your specific precision requirements.

**Step 3. Adversarial Attack**
After obtaining the trained 3DGS model, follow these steps to start the attack.

Prepare the configuration files and sample data as demonstrated in [DEMO](https://pan.baidu.com/s/1UiRpV4LAoDC8rLi2cF0ECw) pwsd: rpga.

Adjust the hyperparameters in the command-line arguments or config files.

Run the training script:

`python train_rel_attack_hpcm.py`
