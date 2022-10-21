# explainable-ai
We used explainable AI techniques to identify signatures of flare triggering events by training high performance ConvNets on the binary-classification task of distinguishing Mg II spectra generated from active regions that lead to a solar flare and those that did not. After achieving high TSS scores we automatically identified the important features of the positive class using the [Grad-CAM](https://arxiv.org/abs/1610.02391) and [Expected gradients](https://arxiv.org/abs/1906.10670) formalism to generate saliency maps that highlighted the discriminant regions of spectrograms (audio type signals that encode the physics of the solar atmosphere). Grad-CAM leverages the spatial coherency and complex pattern recognition of the final convolutional layers of our network, forming a coarse weighted sum of the most important features. The activations of these feature maps are amplified for patterns that affect the prediction most, and projected back to the resolution of the input spectrum generating a saliency map. Expected gradients on the other hand works at the resolution of the input, using the game theoretic idea of missingness and Shapley values to generate saliency maps. For Expected gradients we used a modified version of the code from the following [Git](https://github.com/suinleelab/attributionpriors) repository. 
