#include "water_color.h"

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
                                      cv::cuda::PtrStepSz<char> markers,
                                      int rows, int cols)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < markers.cols && y < markers.rows)
      {
            if(markers.ptr(y)[x] == -1)
            {
                  int diff = 255 * 3;
                  uchar val[3] = {diff_image_b.ptr(y)[x],
                                  diff_image_g.ptr(y)[x],
                                  diff_image_r.ptr(y)[x]};
                  char lab = -1;

                  for (int i = -1; i <= 1; i++)
                  {
                        for (int j = -1; j <= 1; j++)
                        {
                              if (cudaIsInBounds(cols, rows, x + i, y + j))
                              {
                                    char temp_lab = markers.ptr(y+j)[x+i];
                                    if (temp_lab > -1)
                                    {
                                          int temp_diff = abs(val[0] - diff_image_b.ptr(y+j)[x+i]) +
                                                          abs(val[1] - diff_image_g.ptr(y+j)[x+i]) +
                                                          abs(val[2] - diff_image_r.ptr(y+j)[x+i]);
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


void post_water_seg(cv::cuda::GpuMat d_diff_image, cv::cuda::PtrStepSz<char> markers)
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

      cudaGetLastError();
      cudaDeviceSynchronize();
}
