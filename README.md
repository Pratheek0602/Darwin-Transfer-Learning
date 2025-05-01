Enhanced version of the ITUNet framework based on [Swangeese Team's implementation](https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge) for the PICAI 2022 Challenge.

## Enhancements Over Original Implementation

This repository extends the original work with the following improvements:

1. **Transfer Learning Integration** 
   - Added loading of pretrained ResNet weights (ResNet18/34/50) for faster convergence and better performance
   - Implemented differential learning rates for pretrained encoder vs. random-initialized decoder components

2. **Advanced Weight Initialization**
   - Implemented He/Kaiming initialization for convolutional layers
   - Applied Xavier initialization for linear layers
   - Added conservative weight scaling to prevent numerical instability
  
## How to Run

1. Replace the `segmentation` folder in the original Swangeese repository with this folder.
2. Follow the instructions in the [Swangeese README](https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge) for setup and execution.
