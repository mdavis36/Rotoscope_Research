# Rotoscope_Research
Project code for IEEE paper "GPU Acceleration of a Best-Features Based Digital Rotoscope". 

Abstract-- This paper presents the first hybrid GPU-CPU implementation of a best-features based digital rotoscope. Unlike the other rotoscoping and Non Photo-Realistic Rendering (NPR) implementations, this best-features rotoscoping technique incorporates four major stages: background subtraction, best-feature corner detection, marker-based watershed segmentation, and color palette to produce a cartoon stylized video sequence. Our GPU-based implementation uses the computing power of both the CPU host and the GPU device for fast processing. We implement the computationally-intensive and parallel stages on the GPU device by using optimizations such as shared memory for reduced global memory accesses and execution configuration to maximize the GPU multiprocessor occupancy. We also devise a novel window-based reduction method on the GPU device to select optimally spaced best features; this task is inherently sequential in the serial algorithm. We test our implementation using videos of different resolutions. Our GPU implementation reports speedup as high as 3.71x for a video resolution of 1440 x 2560, and an end-to-end execution time reduction from 21 minutes to 6 minutes for the largest video of resolution 2160 x 3840.