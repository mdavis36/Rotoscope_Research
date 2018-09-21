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
            float x_sum = 0;
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
                                           float *lambda,
                                           int width,
                                           int height)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;

      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int indx = tiy * width + tix;

            int half_width = (CONVOL_WIDTH - 1) / 2;

            float G_x, G_y;
            float G_xy;
            float G_x2 = 0.0f;
            float G_y2 = 0.0f;
            int con_indx;

            for (int x = -half_width; x <= half_width; x++)
            {
                  for (int y = -half_width; y <= half_width; y++)
                  {
                        if (cudaIsInBounds(width, height, tix + x, tiy + y)) con_indx = (tiy + y) * width + (tix + x);
                        else con_indx = indx;
                        G_x = deriv_x[con_indx];
                        G_y = deriv_y[con_indx];

                        G_x2 += G_x * G_x;
                        G_y2 += G_y * G_y;
                        G_xy += G_x * G_y;
                  }
            }

            float D = G_x2 + G_y2;
            float E = D / 2;
            float F = sqrtf(D*D - 4 * (G_x2 * G_y2 - G_xy * G_xy));

            float l_1 = E + F;
            float l_2 = E - F;

            lambda[indx] = min(l_1, l_2);
            //cornerness[indx] = min(l_1, l_2);
            cornerness[indx] = (G_x2 * G_y2) - (G_xy * G_xy) - 0.04 * (G_x2+G_y2) * (G_x2+G_y2);
            //cornerness[indx] = G_x2 + G_x2 - sqrtf((G_x2 - G_y2) * (G_x2 - G_y2) + (G_xy) * (G_xy)); //Used from eq in gfft.cu opencv
            indexes[indx] = indx;
      }

}


__global__ void reduceEigenData(float *cornerness,
                                float *indexes,
                                float *o_cornerness,
                                float *o_indexes,
                                int width,
                                int height,
                                int w_size)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;
      int tindx = tiy * (blockDim.x * gridDim.x) + tix;

      int imx = tix * w_size;
      int imy = tiy * w_size;

      int temp_indx = imy * width + imx;
      float max_cornerness = cornerness[temp_indx];
      int max_index = temp_indx;

      for (int y = 0; y < w_size; y++)
      {
            for(int x = 0; x < w_size; x++)
            {
                  if (cudaIsInBounds(width, height, imx + x, imy + y))
                  {
                        temp_indx = (imy + y) * width + (imx + x);
                        if (max_cornerness < cornerness[temp_indx])
                        {
                              max_cornerness = cornerness[temp_indx];
                              max_index = temp_indx;
                        }
                  }
            }
      }
      o_cornerness[tindx] = max_cornerness;
      o_indexes[tindx] = max_index;
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

void rotoscope(unsigned char *h_img,
               unsigned char *h_back,
               unsigned char *h_diff,
               unsigned char *h_diff_gray,
               unsigned char *h_corners,
               unsigned char *h_smooth_x,
               unsigned char *h_smooth_y,
               unsigned char *h_deriv_x,
               unsigned char *h_deriv_y,
               int width,
               int height,
               size_t _channel_size)
{
      unsigned char *d_img, *d_back, *d_diff, *d_diff_gray;

      float *d_gaus_mask, *d_gaus_deriv_mask;

      unsigned char *d_smooth_x;
      unsigned char *d_smooth_y;
      float *d_deriv_x;
      float *d_deriv_y;

      float *d_cornerness;
      float *d_indexes;
      float *d_lambda;
      float *d_red_cornerness;
      float *d_red_indexes;

      int reduction_window = 5;

      // Generate Gaussian Mask values
      float *gaus, *gaus_d;
      gaus = (float *)malloc(sizeof(float) * MASK_WIDTH);
      gaus_d = (float *)malloc(sizeof(float) * MASK_WIDTH);
      generateGuassianMasks(MASK_WIDTH, gaus, gaus_d);


      // Initialize Timer
      float t_1, t_2;


      // Alocate Device Memory for Rotoscoping
      gpuErrchk( cudaMalloc(&d_img,       _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_back,      _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_diff,      _channel_size * 3) );
      gpuErrchk( cudaMalloc(&d_diff_gray, _channel_size) );

      // Device Memory for Masks
      gpuErrchk( cudaMalloc(&d_gaus_mask,       MASK_WIDTH * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_gaus_deriv_mask, MASK_WIDTH * sizeof(float)) );

      // Device Memory for gaussian and Derivative Images
      gpuErrchk( cudaMalloc(&d_smooth_x, _channel_size) );
      gpuErrchk( cudaMalloc(&d_smooth_y, _channel_size) );
      gpuErrchk( cudaMalloc(&d_deriv_x,  width * height * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_deriv_y,  width * height * sizeof(float)) );

      // Device Memory for feature Output
      gpuErrchk( cudaMalloc(&d_cornerness, width * height * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_indexes,    width * height * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_lambda,     width * height * sizeof(float)) );

      gpuErrchk( cudaMalloc(&d_red_cornerness, std::ceil(width /reduction_window) * std::ceil(height /reduction_window) * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_red_indexes,    std::ceil(width /reduction_window) * std::ceil(height /reduction_window) * sizeof(float)) );


      // Copying host input data to d_img and d_back
      gpuErrchk( cudaMemcpy(d_img,  h_img,  _channel_size * 3, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_back, h_back, _channel_size * 3, cudaMemcpyHostToDevice) );

      // Copying Mask data to Device
      gpuErrchk( cudaMemcpy(d_gaus_mask,       gaus,   MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_gaus_deriv_mask, gaus_d, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice) );



      // Calculate block and grid dimensions
      dim3 block(32,16);
      dim3 grid(std::ceil((float)width / block.x), std::ceil((float)height / block.y));

      // CUDA kernel calls
      diff_and_convert_kernel<<<grid, block>>>(d_img, d_back, d_diff, d_diff_gray, width, height);
      t_1 = getProcessTime();
      smooth_image_kernel<<<grid, block>>>(d_diff_gray, d_smooth_x, d_smooth_y, d_gaus_mask, width, height);
      derive_image_kernel<<<grid, block>>>(d_smooth_x, d_smooth_y, d_deriv_x, d_deriv_y, d_gaus_deriv_mask, width, height);
      compute_eigenvalues_kernel<<<grid, block>>>(d_deriv_x, d_deriv_y, d_cornerness, d_indexes, d_lambda, width, height);


      //grid = dim3(((width / reduction_window) / block.x), ((height / reduction_window) / block.y));
      //reduceEigenData<<<grid, block>>>(d_cornerness, d_indexes, d_red_cornerness, d_red_indexes, width, height, reduction_window);



      // Create Thrust device memory pointers for sorting
      thrust::device_ptr<float> t_cornerness = thrust::device_pointer_cast(d_cornerness);
      thrust::device_ptr<float> t_indexes    = thrust::device_pointer_cast(d_indexes);
      //thrust::device_ptr<float> t_red_cornerness = thrust::device_pointer_cast(d_red_cornerness);
      //thrust::device_ptr<float> t_red_indexes    = thrust::device_pointer_cast(d_red_indexes);
      //thrust::device_ptr<float> t_lambda    = thrust::device_pointer_cast(d_lambda);
      thrust::sort_by_key(t_cornerness, t_cornerness + (width * height), t_indexes, thrust::greater<float>());
      //thrust::sort_by_key(t_red_cornerness, t_red_cornerness + (std::ceil(width /reduction_window) * std::ceil(height /reduction_window)), t_red_indexes, thrust::greater<float>());
      //thrust::sort_by_key(t_lambda, t_lambda + (width * height), t_indexes, thrust::greater<float>());



      // Alocate memory for feature detection output.
      float *h_cornerness, *h_indexes;
      h_cornerness = (float *)malloc(sizeof(float) * width * height);
      h_indexes =    (float *)malloc(sizeof(float) * width * height);

      float *h_red_cornerness, *h_red_indexes;
      h_red_cornerness = (float *)malloc(sizeof(float) * std::ceil(width /reduction_window) * std::ceil(height /reduction_window));
      h_red_indexes =    (float *)malloc(sizeof(float) * std::ceil(width /reduction_window) * std::ceil(height /reduction_window));

      // Copy Output data back to Host
      gpuErrchk( cudaMemcpy(h_cornerness, d_cornerness, sizeof(float) * width * height, cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_indexes,    d_indexes,    sizeof(float) * width * height, cudaMemcpyDeviceToHost) );


      // -----------------------------------------------------------------------
      //                      Perform Best Corners
      // -----------------------------------------------------------------------

      int *best_corners;
      int num_corners = 1000;
      best_corners = (int *)malloc(sizeof(int) * num_corners);

      bool close;
      int count = 1;
      int curr_indx = 1;
      float quality_level = 0.000001;
      best_corners[0] = (int)h_indexes[0];
      int quality_threshold = h_cornerness[0] * quality_level;

      while(count < num_corners && curr_indx < width * height)
      {
            close = false;
            for (int i = 0; i < count; i++)
            {
                  if (checkInDist(best_corners[i], (int)h_indexes[curr_indx], 4, width)) close = true;
            }

            if (!close && h_cornerness[curr_indx] >= quality_threshold)
            {
                  best_corners[count] = (int)h_indexes[curr_indx];
                  count++;
            }

            curr_indx++;
      }

      // -----------------------------------------------------------------------
      t_2 = getProcessTime();

      gpuErrchk( cudaMemcpy(h_diff,       d_diff,       _channel_size * 3,              cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_diff_gray,  d_diff_gray,  _channel_size,                  cudaMemcpyDeviceToHost) );

      gpuErrchk( cudaMemcpy(h_red_cornerness, d_red_cornerness, sizeof(float) * std::ceil(width /reduction_window) * std::ceil(height /reduction_window), cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_red_indexes,    d_red_indexes,    sizeof(float) * std::ceil(width /reduction_window) * std::ceil(height /reduction_window), cudaMemcpyDeviceToHost) );

      // Copy Debug data back to Host
      gpuErrchk( cudaMemcpy(h_smooth_x,  d_smooth_x,  _channel_size,  cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_smooth_y,  d_smooth_y,  _channel_size,  cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_deriv_x,   d_deriv_x,   _channel_size,  cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(h_deriv_y,   d_deriv_y,   _channel_size,  cudaMemcpyDeviceToHost) );









      std::cout << count << std::endl;

      for (int i = 0; i < num_corners; i++)
      {
            h_corners[(int)best_corners[i]] = 255;
            //h_corners[(int)h_red_indexes[i]] = 255;

            // h_img[(int)best_corners[i]*3] = 255;
            // h_img[(int)best_corners[i]*3+1] = 255;
            // h_img[(int)best_corners[i]*3+2] = 255;
      }
      // for (int i = 0; i < std::ceil(width /reduction_window) * std::ceil(height /reduction_window); i++)
      // {
      //       std::cout << (int)h_red_indexes[i] << std::endl;
      //       //h_corners[(int)h_red_indexes[i]] = 255;
      // }

      gpuErrchk( cudaFree(d_img) );
      gpuErrchk( cudaFree(d_back) );
      gpuErrchk( cudaFree(d_diff) );
      gpuErrchk( cudaFree(d_diff_gray) );


      for (int i = 0; i < MASK_WIDTH; i++)
      {
            std::cout << gaus[i] << ", ";
      }
      std::cout << std::endl;

      for (int i = 0; i < MASK_WIDTH; i++)
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
