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

double getProcessTime()
{
    clock_t t;
    t = clock();
    return ((double)t)/CLOCKS_PER_SEC;
}

__constant__ float c_gaus_mask[MASK_WIDTH];
__constant__ float c_gaus_deriv_mask[MASK_WIDTH];

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

            b = (unsigned char)abs(img[indx3] - back[indx3]);
            g = (unsigned char)abs(img[indx3 + 1] - back[indx3 + 1]);
            r = (unsigned char)abs(img[indx3 + 2] - back[indx3 + 2]);

            o_diff_gray[indx] = (unsigned char)(b * 0.1141 + g * 0.587 + r * 0.2989);
            o_diff[indx3]   = b;
            o_diff[indx3+1] = g;
            o_diff[indx3+2] = r;
      }
}

__global__ void smooth_image_kernel(unsigned char *img,
                                    unsigned char *o_smooth_x,
                                    unsigned char *o_smooth_y,
                                    float *mask,
                                    int width,
                                    int height)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;

      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int offset, i, off_x, off_y, indx;
            float x_sum = 0.001;
            float y_sum = 0;

            float weight;

            indx = tiy * width + tix;

            for (i = 0; i < MASK_WIDTH; i++)
            {
                  offset = i - ((MASK_WIDTH - 1) / 2);
                  off_x = tiy * width + (tix + offset);
                  off_y = (tiy + offset) * width + tix;

                  weight = mask[i];

                  if(cudaIsInBounds(width, height, tix + offset, tiy)) x_sum += weight * img[off_x];
                  else x_sum += weight * img[indx];

                  if(cudaIsInBounds(width, height, tix, tiy + offset)) y_sum += weight * img[off_y];
                  else y_sum += weight * img[indx];
            }

            o_smooth_x[indx] = x_sum;
            o_smooth_y[indx] = y_sum;
      }
}

__global__ void derive_image_kernel(unsigned char *img_x,
                                    unsigned char *img_y,
                                    float *o_derive_x,
                                    float *o_derive_y,
                                    float *mask,
                                    int width,
                                    int height)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;

      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int offset, i, off_x, off_y, indx;
            float x_sum = 0.0f;
            float y_sum = 0.0f;

            float weight;

            indx = tiy * width + tix;

            for (i = 0; i < MASK_WIDTH; i++)
            {
                  offset = i - ((MASK_WIDTH - 1) / 2);
                  off_x = tiy * width + (tix + offset);
                  off_y = (tiy + offset) * width + tix;

                  weight = mask[i];

                  if(cudaIsInBounds(width, height, tix + offset, tiy)) x_sum += weight * img_x[off_x];
                  else x_sum += weight * (float)img_x[indx];

                  if(cudaIsInBounds(width, height, tix, tiy + offset)) y_sum += weight * img_y[off_y];
                  else y_sum += weight * img_y[indx];
            }

            o_derive_x[indx] = x_sum;
            o_derive_y[indx] = y_sum;
      }
}

__global__ void compute_eigenvalues_kernel(float *deriv_x,
                                      float *deriv_y,
                                      float *cornerness,
                                      float *indexes,
                                      int width,
                                      int height)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;

      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int indx = tiy * width + tix;
            //float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            int half_width = (CONVOL_WIDTH - 1) / 2;

            float G_x, G_y; //float G_xy
            float G_x2 = 0.0f;
            float G_y2 = 0.0f;
            float l_1, l_2;
            int con_indx;

            for (int x = -half_width; x <= half_width; x++)
            {
                  for (int y = -half_width; y <= half_width; y++)
                  {
                        if (cudaIsInBounds(width, height, tix + x, tiy + y)) con_indx = (tiy + y) * width + (tix + x);
                        else con_indx = indx;
                        G_x = deriv_x[con_indx];
                        G_y = deriv_y[con_indx];

                        // z[0] += G_x * G_x;
                        // z[1] += G_x * G_y;
                        // z[2] += G_y * G_y;
                        // z[3] += G_y * G_y;

                        G_x2 += G_x * G_x;
                        G_y2 += G_y * G_y;
                        //G_xy+=G_x*G_y;
                  }
            }

            float D = (G_x2 + G_y2);
            D *= D;

            l_1 = (D/4) + D + (4 * G_x2 * G_y2);
            l_2 = (D/4) - D + (4 * G_x2 * G_y2);

            cornerness[indx] = l_1 * l_2 - K_VAL * (l_1 + l_2) * (l_1 + l_2);
            //cornerness[indx] = G_x2*G_y2 - G_xy*G_xy - 0.04(G_x2+G_y2)^2;
            //cornerness[indx] = min(l_1, l_2);
            indexes[indx] = indx;
      }

      //thrust::sort_by_key(thrust::device, cornerness, cornerness + (width * height), indexes, thrust::greater<int>());
}

bool checkInDist(int ind1, int ind2, int dist, int width)
{
      int x1, x2, y1, y2, a, b;

      x1 = ind1 % width;
      y1 = ind1 / width;

      x2 = ind2 % width;
      y2 = ind2 / width;

      a = abs(x1 - x2);
      b = abs(y1 - y2);

      int a2b2 = (a*a) + (b*b);
      if ((dist*dist) >  a2b2) return true;
      return false;
}

void diff_and_convert(unsigned char *h_img,
                      unsigned char *h_back,
                      unsigned char *h_diff,
                      unsigned char *h_diff_gray,
                      int width,
                      int height,
                      size_t _channel_size)
{
      unsigned char *d_img, *d_back, *d_diff, *d_diff_gray;

      unsigned char *d_smooth_x;
      unsigned char *d_smooth_y;

      float *d_gaus_mask, *d_gaus_deriv_mask;

      float *d_deriv_x;
      float *d_deriv_y;

      //unsigned char *d_lambda_1, *d_lambda_2;
      float *d_cornerness;
      float *d_indexes;

      int w = MASK_WIDTH;
      float *gaus, *gaus_d;
      gaus = (float *)malloc(sizeof(float) * w);
      gaus_d = (float *)malloc(sizeof(float) * w);
      generateGuassianMasks(w, gaus, gaus_d);



      float t_1, t_2;
      t_1 = getProcessTime();



      gpuErrchk( cudaMalloc(&d_img,  _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_back, _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_diff, _channel_size * 3) );

      gpuErrchk( cudaMalloc(&d_diff_gray, _channel_size) );
      gpuErrchk( cudaMalloc(&d_smooth_x, _channel_size) );
      gpuErrchk( cudaMalloc(&d_smooth_y, _channel_size) );

      gpuErrchk( cudaMalloc(&d_gaus_mask, MASK_WIDTH * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_gaus_deriv_mask, MASK_WIDTH * sizeof(float)) );

      gpuErrchk( cudaMalloc(&d_deriv_x, width * height * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_deriv_y, width * height * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_cornerness, width * height * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_indexes, width * height * sizeof(int)) );

      gpuErrchk( cudaMemcpy(d_img,  h_img,  _channel_size * 3, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_back, h_back, _channel_size * 3, cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMemcpy(d_gaus_mask, gaus, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_gaus_deriv_mask, gaus_d, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice) );

      //gpuErrchk( cudaMemcpyToSymbol(c_gaus_mask,       &gaus,   MASK_WIDTH * sizeof(float)) );
      //gpuErrchk( cudaMemcpyToSymbol(c_gaus_deriv_mask, &gaus_d, MASK_WIDTH * sizeof(float)) );

      dim3 block(32,16);
      dim3 grid(std::ceil((float)width / block.x), std::ceil((float)height / block.y));

      diff_and_convert_kernel<<<grid, block>>>(d_img, d_back, d_diff, d_diff_gray, width, height);
      smooth_image_kernel<<<grid, block>>>(d_diff_gray, d_smooth_x, d_smooth_y, d_gaus_mask, width, height);
      derive_image_kernel<<<grid, block>>>(d_smooth_x, d_smooth_y, d_deriv_x, d_deriv_y, d_gaus_deriv_mask, width, height);
      compute_eigenvalues_kernel<<<grid, block>>>(d_deriv_x, d_deriv_y, d_cornerness, d_indexes, width, height);

      thrust::device_ptr<float> t_cornerness = thrust::device_pointer_cast(d_cornerness);
      thrust::device_ptr<float> t_indexes    = thrust::device_pointer_cast(d_indexes);
      thrust::sort_by_key(t_cornerness, t_cornerness + (width * height), t_indexes, thrust::greater<float>());
      //thrust::sort_by_key(thrust::device, t_cornerness, t_cornerness + (width * height), t_indexes, thrust::greater<int>());


      float *h_cornerness, *h_indexes;
      h_cornerness = (float *)malloc(sizeof(float) * width * height);
      h_indexes = (float *)malloc(sizeof(float) * width * height);
      gpuErrchk( cudaMemcpy(h_cornerness, d_cornerness, sizeof(float) * width * height, cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_indexes, d_indexes, sizeof(float) * width * height, cudaMemcpyDeviceToHost) );

      //thrust::sort_by_key(thrust::host, h_cornerness, h_cornerness + (width * height), h_indexes, thrust::greater<float>());

      t_2 = getProcessTime();

      gpuErrchk( cudaMemcpy(h_diff,      d_diff,      _channel_size * 3, cudaMemcpyDeviceToHost) );
      //gpuErrchk( cudaMemcpy(h_diff_gray, d_diff_gray, _channel_size    , cudaMemcpyDeviceToHost) );

      for(int i = 0; i < 1000; i++)
      {
            h_diff_gray[(int)h_indexes[i]] = 255;
            std::cout << h_cornerness[i] << std::endl;

      }

      // unsigned char *h_smooth_x, *h_smooth_y;
      // h_smooth_x = (unsigned char *)malloc(_channel_size);
      // h_smooth_y = (unsigned char *)malloc(_channel_size);
      // gpuErrchk( cudaMemcpy(h_smooth_x, d_smooth_x, _channel_size, cudaMemcpyDeviceToHost) );
      // gpuErrchk( cudaMemcpy(h_smooth_y, d_smooth_y, _channel_size, cudaMemcpyDeviceToHost) );
      //
      // for (int  i = 0; i < 50; i ++)
      // {
      //       std::cout << (int)h_smooth_x[i] << ", " << (int)h_smooth_y[i] << std::endl;
      // }
      //
      // float *h_deriv_x, *h_deriv_y;
      // h_deriv_x = (float *)malloc(sizeof(float) * width * height);
      // h_deriv_y = (float *)malloc(sizeof(float) * width * height);
      // gpuErrchk( cudaMemcpy(h_deriv_x, d_deriv_x, sizeof(float) * width * height, cudaMemcpyDeviceToHost) );
      // gpuErrchk( cudaMemcpy(h_deriv_y, d_deriv_y, sizeof(float) * width * height, cudaMemcpyDeviceToHost) );
      //
      // for (int  i = 0; i < 50; i ++)
      // {
      //       std::cout << h_deriv_x[i] << ", " << h_deriv_y[i] << std::endl;
      // }

      gpuErrchk( cudaFree(d_img) );
      gpuErrchk( cudaFree(d_back) );
      gpuErrchk( cudaFree(d_diff) );
      gpuErrchk( cudaFree(d_diff_gray) );


      for (int i = 0; i < w; i++)
      {
            std::cout << gaus[i] << ", ";
      }
      std::cout << std::endl;

      for (int i = 0; i < w; i++)
      {
            std::cout << gaus_d[i] << ", ";
      }
      std::cout << std::endl;
      std::cout << t_2 - t_1 << " Seconds" << std::endl;
}

void generateGuassianMasks(int w, float * g, float * g_d)
{
      int a = (w - 1) / 2;
      float sigma = (a + 0.5) / 2.5;
      float sig_2 = sigma * sigma;

      int i, b;
      float sum_g = 0.0f;
      float sum_g_d = 0.0f;

      for (i = 0; i < w; i++)
      {
                  b = i-a;
                  g[i] = exp( (-1 * b * b) / (2 * sig_2) );
                  g_d[i] = -1 * b * g[i];
                  sum_g += g[i];
                  sum_g_d += -i * g_d[i];
      }

      for (i = 0; i < w; i++)
      {
            g[i] /= sum_g;
            g_d[i] /= -sum_g_d;
      }
}
