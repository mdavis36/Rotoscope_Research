#ifndef WATER_COLOR_H
#define WATER_COLOR_H

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

void do_test(cv::cuda::PtrStepSz<uchar> in, cv::cuda::PtrStepSz<uchar> out);
void post_water_seg(cv::cuda::GpuMat d_diff_image, cv::cuda::PtrStepSz<int> markers);
void color_palette_image(cv::cuda::GpuMat d_image,
                         cv::cuda::GpuMat markers,
                         cv::cuda::GpuMat colors,
                         cv::cuda::GpuMat out);

#endif
