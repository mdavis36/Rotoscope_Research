#include "Roto.h"


// -----------------------------------------------------------------------------
// Function Name : gpuErrchk, gpuAssert
//
// Description : CUDA call error checking and handling. This C function was copied
//               from stack overflows website from user talonmies comment.
//
// Credit :
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
// -----------------------------------------------------------------------------
extern "C" {
      #include <stdio.h>
      #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
      inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
      {
         if (code != cudaSuccess)
         {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
         }
      }
}


__device__ bool cudaIsInBounds(int width, int height, int x, int y)
{
      return x < width && x >= 0 && y < height && y >=0 ? true : false;
}

__global__ void diff_and_convert_kernel(unsigned char *img,
                                        unsigned char *back,
                                        unsigned char *o_diff,
                                        unsigned char *o_diff_gray,
                                        int width,
                                        int height)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;

      if (cudaIsInBounds(width, height, tix, tiy))
      {
            unsigned char b,g,r;

            int indx = tiy * width + tix;
            int indx3 = indx * 3;

            b = abs(img[indx3] - back[indx3]);
            g = abs(img[indx3 + 1] - back[indx3 + 1]);
            r = abs(img[indx3 + 2] - back[indx3 + 2]);
            o_diff_gray[indx] = b * 0.1141 + g * 0.587 + r * 0.2989;
      }
}

void diff_and_convert(unsigned char *img,
                      unsigned char *back,
                      unsigned char*o_diff,
                      unsigned char *o_diff_gray,
                      int width,
                      int height)
{
      unsigned char *d_img, *d_back, *d_diff, *d_diff_gray;

      size_t _channel_size = sizeof(unsigned char) * width * height;

      gpuErrchk( cudaMalloc(&d_img,  _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_back, _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_diff, _channel_size * 3) );

      gpuErrchk( cudaMalloc(&d_diff_gray, _channel_size) );

      gpuErrchk( cudaMemcpy(d_img,  img,  _channel_size * 3, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_back, back, _channel_size * 3, cudaMemcpyHostToDevice) );

      dim3 block(32,16);
      dim3 grid(std::ceil((float)width / block.x), std::ceil((float)height / block.y));

      diff_and_convert_kernel<<<grid, block>>>(d_img, d_back, d_diff, d_diff_gray, width, height);

      //gpuErrchk( cudaMemcpy(o_diff,      d_diff,      _channel_size * 3, cudaMemcpyDeviceToHost) );
      //gpuErrchk( cudaMemcpy(o_diff_gray, d_diff_gray, _channel_size    , cudaMemcpyDeviceToHost) );

      gpuErrchk( cudaFree(d_img) );
      gpuErrchk( cudaFree(d_back) );
      gpuErrchk( cudaFree(d_diff) );
      gpuErrchk( cudaFree(d_diff_gray) );

}
