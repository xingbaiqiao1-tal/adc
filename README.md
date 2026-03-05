# adc

Standard convolutions calculate the dot product between a kernel and a local patch. While effective, this linear operation often requires deep stacking and non-linear activation functions (like ReLU) to capture complex patterns. We propose a "fused" operator that embeds non-linearity directly into the spatial aggregation step. By calculating the absolute difference between the center pixel and its neighbors, the network explicitly learns local variations (gradients).




<img width="1124" height="677" alt="image" src="https://github.com/user-attachments/assets/061658d1-084a-4b93-a153-28922b23aec6" />
