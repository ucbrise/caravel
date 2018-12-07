Paper repo for "Dynamic Space-Time Scheduling for GPU Inference"
-----------------------------------------

## Paper and Posters
[Paper](https://github.com/ucbrise/caravel/raw/master/assets/paper.pdf)

[Poster](https://github.com/ucbrise/caravel/raw/master/assets/poster.pdf)

## Abstract
Serving deep neural networks in latency critical interactive settings often requires GPU acceleration. However, the small batch sizes typical in online inference results in poor GPU utilization, a potential performance gap which GPU resource sharing can address.

In this paper, we explore several techniques to leverage both temporal and spatial multiplexing to improve GPU utilization for deep learning inference workloads. We evaluate the performance trade-offs of each approach with respect to resource-efficiency, latency predictability, and isolation when compared with conventional batched inference.

Our experimental analysis suggests up to a 5x potential for improved utilization through the exploration of more advanced spatial and temporal multiplexing strategies. Our preliminary prototype of a dynamic space-time scheduler demonstrates a 3.23x floating-point throughput increase over space-only multiplexing and a 7.73x increase over time-only multiplexing for convolutions, while also providing better isolation and latency predictability.


