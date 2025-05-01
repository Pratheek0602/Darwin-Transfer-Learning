Enhanced version of the ITUNet framework based on [SwanGeese Team's implementation](https://github.com/Yukiya-Umimi/ITUNet-for-PICAI-2022-Challenge) for the PICAI 2022 Challenge.

## Enhancements Over Original Implementation

This repository extends the original work with the following improvements:

1. **Transfer Learning Integration** 
   - Added loading of pretrained ResNet weights (ResNet18/34/50) for faster convergence and better performance
   - Implemented differential learning rates for pretrained encoder vs. random-initialized decoder components

2. **Advanced Weight Initialization**
   - Implemented He/Kaiming initialization for convolutional layers
   - Applied Xavier initialization for linear layers
   - Added conservative weight scaling to prevent numerical instability