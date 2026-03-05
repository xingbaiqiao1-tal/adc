# adc
adc net
Standard convolutions calculate the dot product between a kernel and a local patch. While effective, this linear operation often requires deep stacking and non-linear activation functions (like ReLU) to capture complex patterns. We propose a "fused" operator that embeds non-linearity directly into the spatial aggregation step. By calculating the absolute difference between the center pixel and its neighbors, the network explicitly learns local variations (gradients).

2.1 The AbsDiff-Conv FormulationLet $x$ be the center pixel of a sliding window, $x_i$ be the $i$-th neighbor, and $w_i$ be the learnable weight. The output $y$ is defined as:$$y = \sum_{i \in \Omega} (w_i \cdot |x - x_i|) + \sum_{i \in \Omega} (w_i \cdot x_i)$$Where $\Omega$ represents the local neighborhood (e.g., $3 \times 3$).

2.2 Mathematical IntuitionLinear Component ($\sum w_i x_i$): Acts as a standard feature extractor, preserving the fundamental properties of CNNs.Non-linear Component ($\sum w_i |x - x_i|$): Measures the "distance" in the intensity space. This term acts as an adaptive edge detector. If $w_i > 0$, the network rewards high contrast; if $w_i < 0$, it penalizes sharp changes (smoothing).
