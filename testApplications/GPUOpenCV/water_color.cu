#include "water_color.h"


#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


__global__ void do_test_kernel(cv::cuda::PtrStepSz<uchar> in, cv::cuda::PtrStepSz<uchar> out)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < out.cols && y < out.rows)
      {
            out.ptr(in.rows - y)[x] = in.ptr(y)[x];
      }

}


void do_test(cv::cuda::PtrStepSz<uchar> in, cv::cuda::PtrStepSz<uchar> out)
{
      dim3 block(32,8);
      dim3 grid(cv::divUp(in.cols, block.x), cv::divUp(out.rows, block.y));

      do_test_kernel<<<grid,block>>>(in,out);

      cudaGetLastError();
      cudaDeviceSynchronize();
}

__device__ bool cudaIsInBounds(int width, int height, int x, int y)
{
      return x < width && x >= 0 && y < height && y >=0 ? true : false;
}

__global__ void post_water_seg_kernel(cv::cuda::PtrStepSz<uchar> diff_image_b,
                                      cv::cuda::PtrStepSz<uchar> diff_image_g,
                                      cv::cuda::PtrStepSz<uchar> diff_image_r,
                                      cv::cuda::PtrStepSz<int> markers,
                                      int rows, int cols)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < markers.cols && y < markers.rows)
      {
            if(markers.ptr(y)[x] == -1)
            {
                  int diff = 255 * 4;
                  int val[3] = {int(diff_image_b.ptr(y)[x]),
                                int(diff_image_g.ptr(y)[x]),
                                int(diff_image_r.ptr(y)[x])};
                  int lab = -1;
                  int temp_lab = -1;

                  for (int i = -1; i < 2; i++)
                  {
                        for (int j = -1; j < 2; j++)
                        {
                              if (cudaIsInBounds(cols, rows, x + i, y + j) && !(j == 0 && i == 0))
                              {
                                    temp_lab = markers.ptr(y+j)[x+i];
                                    if (temp_lab > -1)
                                    {
                                          int temp_diff = abs(val[0] - int(diff_image_b.ptr(y+j)[x+i])) +
                                                          abs(val[1] - int(diff_image_g.ptr(y+j)[x+i])) +
                                                          abs(val[2] - int(diff_image_r.ptr(y+j)[x+i]));
                                          if (temp_diff < diff)
                                          {
                                                diff = temp_diff;
                                                lab = temp_lab;
                                          }
                                    }
                              }
                        }
                  }
                  markers.ptr(y)[x] = lab;
            }
      }
}


void post_water_seg(cv::cuda::GpuMat d_diff_image, cv::cuda::PtrStepSz<int> markers)
{
      cv::cuda::GpuMat d_diff_channels[3];
      cv::cuda::split(d_diff_image, d_diff_channels);

      dim3 block(32,8);
      dim3 grid(cv::divUp(markers.cols, block.x), cv::divUp(markers.rows, block.y));

      post_water_seg_kernel<<<grid,block>>>(d_diff_channels[0],
                                            d_diff_channels[1],
                                            d_diff_channels[2],
                                            markers,
                                            d_diff_image.rows, d_diff_image.cols);

      cudaCheckError();
      cudaGetLastError();
      cudaDeviceSynchronize();
}



__global__ void color_palette_kernel(cv::cuda::PtrStepSz<uchar> image_b,
                                     cv::cuda::PtrStepSz<uchar> image_g,
                                     cv::cuda::PtrStepSz<uchar> image_r,
                                     cv::cuda::PtrStepSz<int> markers,
                                     cv::cuda::PtrStepSz<int> colors,
                                     int rows, int cols
                                    )
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < markers.cols && y < markers.rows)
      {
            int index = markers.ptr(y)[x];
            if(index != -1)
            {
                  atomicAdd(&colors.ptr(0)[index], int(image_b.ptr(y)[x]));
                  atomicAdd(&colors.ptr(1)[index], int(image_g.ptr(y)[x]));
                  atomicAdd(&colors.ptr(2)[index], int(image_r.ptr(y)[x]));
                  atomicAdd(&colors.ptr(3)[index], 1);
            }
      }
}

__global__ void avg_color_palette(cv::cuda::PtrStepSz<int> colors, int rows, int cols)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < colors.cols && y < colors.rows - 1)
      {
            int count = colors.ptr(3)[x];
            colors.ptr(y)[x] /= count;
      }
}

__global__ void fill_output_image_kernel(cv::cuda::PtrStepSz<int> markers,
                                         cv::cuda::PtrStepSz<int> colors,
                                         cv::cuda::PtrStepSz<uchar> out_b,
                                         cv::cuda::PtrStepSz<uchar> out_g,
                                         cv::cuda::PtrStepSz<uchar> out_r,
                                         int rows, int cols)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < markers.cols && y < markers.rows)
      {
            int index = markers.ptr(y)[x];
            if(index != -1)
            {
                  out_b.ptr(y)[x] = colors.ptr(0)[index];
                  out_g.ptr(y)[x] = colors.ptr(1)[index];
                  out_r.ptr(y)[x] = colors.ptr(2)[index];
            }
      }
}

void color_palette_image(cv::cuda::GpuMat d_image,
                         cv::cuda::GpuMat markers,
                         cv::cuda::GpuMat colors,
                         cv::cuda::GpuMat out)
{
      cv::cuda::GpuMat d_image_channels[3];
      cv::cuda::GpuMat d_out_channels[3];

      cv::cuda::split(d_image, d_image_channels);
      cv::cuda::split(out, d_out_channels);

      dim3 block(32,8);
      dim3 grid(cv::divUp(markers.cols, block.x), cv::divUp(markers.rows, block.y));

      color_palette_kernel<<<grid,block>>>(d_image_channels[0],
                                           d_image_channels[1],
                                           d_image_channels[2],
                                           markers,
                                           colors,
                                           d_image.rows, d_image.cols);
      cudaGetLastError();
      cudaDeviceSynchronize();

      dim3 block_avg(64,3);
      dim3 grid_avg(cv::divUp(colors.cols, block_avg.x), cv::divUp(colors.rows, block_avg.y));
      avg_color_palette<<<grid_avg, block_avg>>>(colors, colors.rows, colors.cols);

      cudaGetLastError();
      cudaDeviceSynchronize();

      fill_output_image_kernel<<<grid,block>>>(markers,
                                               colors,
                                               d_out_channels[0],
                                               d_out_channels[1],
                                               d_out_channels[2],
                                               d_image.rows, d_image.cols);

      cudaGetLastError();
      cudaDeviceSynchronize();

      cv::cuda::merge(d_out_channels, 3, out);

}
