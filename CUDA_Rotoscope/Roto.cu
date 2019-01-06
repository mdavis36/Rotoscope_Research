// -----------------------------------------------------------------------------
// Filename : GPU_gaus.cu
// Last Update : 01/06/2019
// Author   : Michael Davis
//
// Description : .cu CUDA C code designed to perform GPU accelerated Gaussian Blur
//               on 24bit bitmap RGB data.
//
// -----------------------------------------------------------------------------

#include "Roto.cuh"
#include <cub/cub.cuh>
#include "MyCompVision.h"

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


// -----------------------------------------------------------------------------
// Function Name : getSharedGlobalCoOrds
//
// Description : retrieve cartesian coordinate values of a 2D shared memory structure.
//
// Inputs : sindx - relative index in shared memory.
//		bpos  - gloabal 2D coordinates of 2D cuda block.
//		sDim  - Dimensions of 2D shared memory.
//		w, h  - width and height of image.
//		hw	- half width of filter (filterwidth - 1) / 2
//
// Output : out - out co-ordinates relative to the image of the shared memory
//		bool - if the coordinates are within the image or not.
//         
// -----------------------------------------------------------------------------

__device__ bool getSharedGlobalCoOrds(int sindx, dim3 bpos, dim3 sDim, int w, int h, int hw, dim3 *out)
{
	dim3 spos(sindx%sDim.x, sindx/sDim.x);
	dim3 gpos(bpos.x+spos.x-hw, bpos.y+spos.y-hw);
	if (cudaIsInBounds(w, h, gpos.x, gpos.y))
	{
		*out = gpos;
		return true;
	}
	return false;
}


// -----------------------------------------------------------------------------
// Function Name : getSharedIndx
//
// Description : retrieve realtive shared memory index based on 2D global coordinates.
//
// Inputs : gpos  - globaal 2D coordinates of 2D position.
//		bpos  - gloabal 2D coordinates of 2D cuda block.
//		sDim  - Dimensions of 2D shared memory.
//		hw	- half width of filter (filterwidth - 1) / 2
//
// Output : indx - out index relative to the image of the shared memory
//		bool - if the coordinates are within the shared memory dimensions.
//         
// -----------------------------------------------------------------------------
__device__ bool getSharedIndx(dim3 gpos, dim3 bpos, dim3 sDim, int hw, int *indx)  
{
	dim3 spos(gpos.x - bpos.x + hw, gpos.y - bpos.y + hw);
	if (cudaIsInBounds(sDim.x, sDim.y, spos.x, spos.y))
	{
		*indx = spos.y * sDim.x + spos.x;
		return true;
	}
	return false;
}



// -----------------------------------------------------------------------------
// Function Name : diff_and_convert_kernel
//
// Description : takes the difference between current frame and background image
//		     converts the result to grayscale and outputs.
//
// Inputs : img   - input image data 
//		back  - background image data	
//		w, h  - width and height of image.
//
// Output : o_diff- colour difference image	
//		o_diff_gray - grayscale difference image
//         
// -----------------------------------------------------------------------------

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
            //o_diff_gray[indx] = (unsigned char)(img[indx3] * 0.1141 + img[indx3+1] * 0.587 + img[indx3+2] * 0.2989);
            o_diff[indx3]   = b;
            o_diff[indx3+1] = g;
            o_diff[indx3+2] = r;
      }
}



// -----------------------------------------------------------------------------
// Function Name : smooth_image_kernel
//
// Description : perform a gaussian filter in the x and y direction and generate
//			an x and y smoothed images. Leverages collaborative shared 
//			memory loading to promote performance.
//
// Inputs : img   - input image data.
//		mask  - gaussian mask 1D array.
//		w, h  - width and height of image.
//		sharedDim - 2D dimaensions of shared memory.
//
// Output : o_smooth_x - smoothed image in x direction.	
//		o_smooth_y - smoothed image in y direction.
//         
// -----------------------------------------------------------------------------

__global__ void smooth_image_kernel(unsigned char *img,
                                    unsigned char *o_smooth_x,
                                    unsigned char *o_smooth_y,
                                    float *mask,
                                    int width,
                                    int height,
						dim3 sharedDim)
{
      extern __shared__ unsigned char s[];


      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int ind = tiy * width + tix;
	int local_index = threadIdx.y * blockDim.x + threadIdx.x;
	int localArea = blockDim.x * blockDim.y;

	int sharedArea = sharedDim.x * sharedDim.y;
	dim3 bpos(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
	
	int halfMask = (MASK_WIDTH - 1) / 2;

	int i = 0;
	while ((i * localArea) + (local_index) < sharedArea)
	{
		int sindx = (i * localArea) + local_index;
		dim3 gPos;
		if (getSharedGlobalCoOrds(sindx, bpos, sharedDim, width, height, halfMask, &gPos))
		{
			s[sindx] = img[gPos.y * width + gPos.x];
		}
		i++;
	}
	__syncthreads();

	int s_indx, s_offx, s_offy;
      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int offset, i, off_x, off_y, indx;
            float x_sum = 0;
            float y_sum = 0;

            float weight;

            indx = tiy * width + tix;
		s_indx = getSharedIndx(dim3(tix, tiy), bpos, sharedDim, halfMask, &s_indx);	

            for (i = 0; i < MASK_WIDTH; i++)
            {
                  offset = i - ((MASK_WIDTH - 1) / 2);
                  off_x = tiy * width + (tix + offset);
                  off_y = (tiy + offset) * width + tix;

                  weight = mask[i];
			if(getSharedIndx(dim3(tix + offset, tiy), bpos, sharedDim, halfMask, &s_offx)) x_sum += weight * s[s_offx];
                  else x_sum += weight * s[s_indx];

                  if(getSharedIndx(dim3(tix, tiy + offset), bpos, sharedDim, halfMask, &s_offy)) y_sum += weight * s[s_offy];
                  else y_sum += weight * s[s_indx];
/*
                  if(cudaIsInBounds(width, height, tix + offset, tiy)) x_sum += weight * img[off_x];
                  else x_sum += weight * img[s_indx];

                  if(cudaIsInBounds(width, height, tix, tiy + offset)) y_sum += weight * img[off_y];
                  else y_sum += weight * img[s_indx];*/
            }

            o_smooth_x[indx] = x_sum;
            o_smooth_y[indx] = y_sum;
      }
}


// -----------------------------------------------------------------------------
// Function Name : derive_image_kernel 
//
// Description : perform a gaussian filter in the x and y direction and generate
//			an x and y derivative images. Leverages collaborative shared 
//			memory loading to promote performance.
//
// Inputs : img_x   - input image data in the x direction.
//		img_y   - input image data in the y direction.
//		mask  - gaussian mask 1D array.
//		w, h  - width and height of image.
//		sharedDim - 2D dimaensions of shared memory.
//
// Output : o_derive_x - smoothed image in x direction.	
//		o_derive_y - smoothed image in y direction.
//         
// -----------------------------------------------------------------------------

__global__ void derive_image_kernel(unsigned char *img_x,
                                    unsigned char *img_y,
                                    float *o_derive_x,
                                    float *o_derive_y,
                                    float *mask,
                                    int width,
                                    int height,
						dim3 sharedDim)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;
	
	int ind = tiy * width + tix;
	int local_index = threadIdx.y * blockDim.x + threadIdx.x;
	int localArea = blockDim.x * blockDim.y;

	int sharedArea = sharedDim.x * sharedDim.y;
	dim3 bpos(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
	
	int halfMask = (MASK_WIDTH - 1) / 2;

	extern __shared__ unsigned char s[];
	unsigned char *s_x = s;
	unsigned char *s_y = (unsigned char*)&s[sharedArea];

	int i = 0;
	while ((i * localArea) + (local_index) < sharedArea)
	{
		int sindx = (i * localArea) + local_index;
		dim3 gPos;
		if (getSharedGlobalCoOrds(sindx, bpos, sharedDim, width, height, halfMask, &gPos))
		{
			s_x[sindx] = img_x[gPos.y * width + gPos.x];
			s_y[sindx] = img_y[gPos.y * width + gPos.x];
		}
		i++;
	}
	__syncthreads();

	int s_indx, s_offx, s_offy;
      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int offset, i, off_x, off_y, indx;
            float x_sum = 0.0f;
            float y_sum = 0.0f;

            float weight;

            indx = tiy * width + tix;
		s_indx = getSharedIndx(dim3(tix, tiy), bpos, sharedDim, halfMask, &s_indx);	

            for (i = 0; i < MASK_WIDTH; i++)
            {
                  offset = i - ((MASK_WIDTH - 1) / 2);
                  off_x = tiy * width + (tix + offset);
                  off_y = (tiy + offset) * width + tix;

                  weight = mask[i];
			if(getSharedIndx(dim3(tix + offset, tiy), bpos, sharedDim, halfMask, &s_offx)) x_sum += weight * s_x[s_offx];
                  else x_sum += weight * s[s_indx];

                  if(getSharedIndx(dim3(tix, tiy + offset), bpos, sharedDim, halfMask, &s_offy)) y_sum += weight * s_y[s_offy];
                  else y_sum += weight * s[s_indx];

           /*       if(cudaIsInBounds(width, height, tix + offset, tiy)) x_sum += weight * img_x[off_x];
                  else x_sum += weight * (float)img_x[indx];

                  if(cudaIsInBounds(width, height, tix, tiy + offset)) y_sum += weight * img_y[off_y];
                  else y_sum += weight * img_y[indx];*/
            }

            o_derive_x[indx] = x_sum;
            o_derive_y[indx] = y_sum;
      }
}


// -----------------------------------------------------------------------------
// Function Name : compute_eigenvaules_kernel 
//
// Description : generates cornerness value images. Leverages collaborative shared
//			memory loading to promote performance.
//
// Inputs : deriv_x   - input image data in the x direction.
//		deriv_y   - input image data in the y direction.
//		mask  - gaussian mask 1D array.
//		w, h  - width and height of image.
//		sharedDim - 2D dimaensions of shared memory.
//
// Output : cornerness - cornerness image data.
//	 	indexes    - equivalent index list used later toi sort cornerness data.	
//         
// -----------------------------------------------------------------------------

__global__ void compute_eigenvalues_kernel(float *deriv_x,
                                           float *deriv_y,
                                           float *cornerness,
                                           float *indexes,
                                           float *lambda,
                                           int width,
                                           int height,
							 dim3 sharedDim)
{
      int tix = blockIdx.x * blockDim.x + threadIdx.x;
      int tiy = blockIdx.y * blockDim.y + threadIdx.y;

	int local_index = threadIdx.y * blockDim.x + threadIdx.x;
	int localArea = blockDim.x * blockDim.y;

	int sharedArea = sharedDim.x * sharedDim.y;
	dim3 bpos(blockIdx.x * blockDim.x, blockIdx.y * blockDim.y);
	
	int halfMask = (CONVOL_WIDTH - 1) / 2;

	extern __shared__ float sf[];
	float *s_x = sf;
	float *s_y = (float*)&sf[sharedArea];

	int i = 0;
	while ((i * localArea) + (local_index) < sharedArea)
	{
		int sindx = (i * localArea) + local_index;
		dim3 gPos;
		if (getSharedGlobalCoOrds(sindx, bpos, sharedDim, width, height, halfMask, &gPos))
		{
			s_x[sindx] = deriv_x[gPos.y * width + gPos.x];
			s_y[sindx] = deriv_y[gPos.y * width + gPos.x];
		}
		i++;
	}
	__syncthreads();


	int s_indx, s_off;
      if (cudaIsInBounds(width, height, tix, tiy))
      {
            int indx = tiy * width + tix;
		s_indx = getSharedIndx(dim3(tix, tiy), bpos, sharedDim, halfMask, &s_indx);	

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
                        //if (cudaIsInBounds(width, height, tix + x, tiy + y)) con_indx = (tiy + y) * width + (tix + x);
                        //else con_indx = indx;

                        //G_x = deriv_x[con_indx];
                        //G_y = deriv_y[con_indx];

				if (getSharedIndx(dim3(tix + x, tiy + y), bpos, sharedDim, halfMask, &s_off)) con_indx = s_off; 
                        else con_indx = s_indx;

                        G_x = s_x[con_indx];
                        G_y = s_y[con_indx];

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


// -----------------------------------------------------------------------------
// Function Name : reduceEigenData 
//
// Description : window reduction of cornerness values, used to aid in seperating best features. 
//
// Inputs : cornerness - cornerness image data. 
//		indexes - equivilant index list for conrerness values.
//		w, h  - width and height of image.
//		w_size - windowing size to use.
//
// Output : cornerness - output cornerness image data. 
//		indexes - output equivilant index list for conrerness values.
//         
// -----------------------------------------------------------------------------
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


// -----------------------------------------------------------------------------
// Function Name : clearMarkerImage 
//
// Description : Resets data in marker image back to 0. 
//
// Inputs : marker_img - marker img array. 
//		cols, rows  - dimensions of image.
//
// -----------------------------------------------------------------------------
__global__ void clearMarkerImage(int *marker_img,
                                 int cols, int rows)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      int indx = y * cols + x;

      if (x < cols && y < rows)
      {
            marker_img[indx] = 0;
      }
}


// -----------------------------------------------------------------------------
// Function Name : generateMarkerImage 
//
// Description : generates the marker image based on sorted marker data. 
//
// Inputs : indexes - equivilant index list for conrerness values.
//		num_markers - number of user specified markers. 
//
// Output : marker_img - output marker image data. 
//         
// -----------------------------------------------------------------------------

__global__ void generateMarkerImage(float *indexes,
                                    int *marker_img,
                                    int num_markers)
{
      int tindx = blockIdx.x * blockDim.x + threadIdx.x;

      if (tindx < num_markers)
      {
            int pos = indexes[tindx];
            marker_img[pos] = tindx;
      }
}


// -----------------------------------------------------------------------------
// Function Name : post_watershed_seg_kernel 
//
// Description : removes border values that are generated by OpenCV's watershed 
//			function.
//
// Inputs : diff_image - original difference image to help determine whish segment
//		a pixel belongs to.
//		cols, rows  - dimensions of image.
//		markers - marker image data.
//		
// -----------------------------------------------------------------------------

__global__ void post_water_seg_kernel(unsigned char *diff_image,
                                      int *markers,
                                      int rows, int cols)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      int indx = y * cols + x;
      int indx3 = indx * 3;
      int conIndx, conIndx3;

      if (x < cols && y < rows)
      {
            if(markers[indx] == -1)
            {
                  int diff = 255 * 4;
                  int val[3] = {int(diff_image[indx3]),
                                int(diff_image[indx3 + 1]),
                                int(diff_image[indx3 + 2])};
                  int lab = -1;
                  int temp_lab = -1;

                  for (int i = -1; i < 2; i++)
                  {
                        for (int j = -1; j < 2; j++)
                        {
                              if (cudaIsInBounds(cols, rows, x + i, y + j) && !(j == 0 && i == 0))
                              {
                                    conIndx = (y + j) * cols + (x + i);
                                    conIndx3 = conIndx * 3;
                                    temp_lab = markers[conIndx];
                                    if (temp_lab > -1)
                                    {
                                          int temp_diff = abs(val[0] - int(diff_image[conIndx3])) +
                                                          abs(val[1] - int(diff_image[conIndx3 + 1])) +
                                                          abs(val[2] - int(diff_image[conIndx3 + 2]));
                                          if (temp_diff < diff)
                                          {
                                                diff = temp_diff;
                                                lab = temp_lab;
                                          }
                                    }
                              }
                        }
                  }
                  markers[indx] = lab;
            }
      }
}


// -----------------------------------------------------------------------------
// Function Name : color_pal_kernel 
//
// Description : generates the color palette.
//
// Inputs : image - original input image.
//		markers - segmented marker image.
//		num_markers - number of user specified markers. 
//		cols, rows  - dimensions of image.
//
// Output : colors - color palette array.
//         
// -----------------------------------------------------------------------------

__global__ void color_pal_kernel(unsigned char *image,
                                     int *markers,
                                     unsigned int *colors,
                                     int rows, int cols
                                    )
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      int indx = y * cols + x;
      int indx3 = indx * 3;

      if (x < cols && y < rows)
      {
            int c_index = markers[indx] * 4;
            if(c_index != -1)
            {
                  atomicAdd(&colors[c_index    ], image[indx3    ]);
                  atomicAdd(&colors[c_index + 1], image[indx3 + 1]);
                  atomicAdd(&colors[c_index + 2], image[indx3 + 2]);
                  atomicAdd(&colors[c_index + 3], 1);
            }
      }
}


// -----------------------------------------------------------------------------
// Function Name : clear_color_palette 
//
// Description : initializes all color palette data to 0.
//
// Inputs : colors - color palette array.
//		num_markers - number of user specified markers. 
//         
// -----------------------------------------------------------------------------

__global__ void clear_color_palette(unsigned int *colors, int num_markers)
{
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      if (indx < num_markers*4)
      {
            colors[indx] = 0;
      }
}


// -----------------------------------------------------------------------------
// Function Name : avg_color_palette 
//
// Description : average out the color palette values.
//
// Inputs : colors - color palette array.
//		num_markers - number of user specified markers. 
//         
// -----------------------------------------------------------------------------

__global__ void avg_color_palette(unsigned int *colors, int num_markers)
{
      int indx = threadIdx.x + blockIdx.x * blockDim.x;

      if (indx < num_markers * 4 && indx%4 != 3)
      {
            int count = colors[indx + 3 - (indx%4)];
            colors[indx] /= count;
      }
}


// -----------------------------------------------------------------------------
// Function Name : fill_output_image_kernel 
//
// Description : generates the final output image.
//
// Inputs : markers - segmented marker image.
//		colors - color palette array.
//		cols, rows  - dimensions of image.
//         
// Output : out - final output image.
// -----------------------------------------------------------------------------

__global__ void fill_output_image_kernel(int *markers,
                                         unsigned int *colors,
                                         unsigned char *out,
                                         int rows, int cols)
{
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;
      int indx = y * cols + x;

      if (x < cols && y < rows)
      {
            int colIndex = markers[indx] * 4;
            int indx3 = indx * 3;
            if(colIndex != -1)
            {
                  out[indx3] = colors[colIndex];
                  out[indx3 + 1] = colors[colIndex + 1];
                  out[indx3 + 2] = colors[colIndex + 2];
            }
      }
}


// -----------------------------------------------------------------------------
// Function Name : CUDA_Rotoscope 
//
// Description : Constructor for the CUDA_Rotoscope object
//
// Inputs : filename - filename of the input video.
//
// -----------------------------------------------------------------------------

CUDA_Rotoscope::CUDA_Rotoscope(std::string filename)
{
      initVideo(filename);

      generateGaussianMasks();
      allocHostMem();

      allocDevMem();
      cpyInitDevMem();

      initCUBSorting();
      initCUDAStuff();
}

CUDA_Rotoscope::~CUDA_Rotoscope()
{
      delete[] frames;
}


// -----------------------------------------------------------------------------
// Function Name : initVideo 
//
// Description : initialize OpenCV video obejct data.
//
// Inputs : filename - filename of the input video.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::initVideo(std::string filename)
{
      cap = cv::VideoCapture(filename);
      height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
      width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
      frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
      _channel_size = sizeof(unsigned char) * height * width;

      frames = new cv::Mat[frame_count];
      cap >> cv_h_back;
      //cv::namedWindow("Back",  cv::WINDOW_AUTOSIZE); cv::imshow("Back",  cv_h_back);
      //cv::waitKey();
      cap.set(cv::CAP_PROP_POS_FRAMES, 1200);

      std::cout << "Max Frames : " << frame_count << std::endl;

      int codec = cv::VideoWriter::fourcc('M','J','P','G');
      output.open("Output.avi", codec, cap.get(cv::CAP_PROP_FPS), cv::Size(width,height), true);
}


// -----------------------------------------------------------------------------
// Function Name : generateGaussianMasks 
//
// Description : generate Gaussian and Gaussian Derivative masks.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::generateGaussianMasks()
{
      gaus = (float *)malloc(sizeof(float) * MASK_WIDTH);
      gaus_d = (float *)malloc(sizeof(float) * MASK_WIDTH);

      int a = (MASK_WIDTH - 1) / 2;
      float sigma = (a + 0.5) / 2.5;
      float sig_2 = sigma * sigma;

      int i, b;
      float sum_g = 0.0f;
      float sum_g_d = 0.0f;

      for (i = 0; i < MASK_WIDTH; i++)
      {
                  b = i-a;
                  gaus[i] = exp( (-1 * b * b) / (2 * sig_2) );
                  gaus_d[i] = -1 * b * gaus[i];
                  sum_g += gaus[i];
                  sum_g_d += -i * gaus_d[i];
      }

      for (i = 0; i < MASK_WIDTH; i++)
      {
            gaus[i] /= sum_g;
            gaus_d[i] /= -sum_g_d;
      }
}

// -----------------------------------------------------------------------------
// Function Name : allocHostMem 
//
// Description : Allocate pinned host memory for copying data back to CPU. Pinned
//			host data allocation promotes performance of memcpy's especially
//			in asynchonously streamed cuda applications.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::allocHostMem()
{
      for (int i = 0; i < NUM_STREAMS; i++)
      {
            cudaMallocHost( &h_markers[i], sizeof(int) * width * height );
            cudaMallocHost( &h_input[i], _channel_size * 3);
            cudaMallocHost( &h_out[i], _channel_size * 3);
            cudaMallocHost( &h_diff_img[i], _channel_size * 3);
      }
      cudaMallocHost( &h_back, _channel_size * 3);
}


// -----------------------------------------------------------------------------
// Function Name : allocDevMem 
//
// Description : Allocate device memory necessary for rotoscoping on the GPU.
//			Compatible with handling allocation of data for multiple streams.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::allocDevMem()
{
      // Device Memory for constant data
      gpuErrchk( cudaMalloc(&d_gaus_mask,       MASK_WIDTH * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_gaus_deriv_mask, MASK_WIDTH * sizeof(float)) );
      gpuErrchk( cudaMalloc(&d_back,            _channel_size * 3) );


      for (int i = 0; i < NUM_STREAMS; i++)
      {
            // Alocate Device Memory for Rotoscoping
            gpuErrchk( cudaMalloc(&d_img[i],       _channel_size * 3) );
            gpuErrchk( cudaMalloc(&d_diff[i],      _channel_size * 3) );
            gpuErrchk( cudaMalloc(&d_diff_gray[i], _channel_size) );

            // Device Memory for gaussian and Derivative Images
            gpuErrchk( cudaMalloc(&d_smooth_x[i], _channel_size) );
            gpuErrchk( cudaMalloc(&d_smooth_y[i], _channel_size) );
            gpuErrchk( cudaMalloc(&d_deriv_x[i],  width * height * sizeof(float)) );
            gpuErrchk( cudaMalloc(&d_deriv_y[i],  width * height * sizeof(float)) );

            // Device Memory for feature Output
            gpuErrchk( cudaMalloc(&d_cornerness[i], width * height * sizeof(float)) );
            gpuErrchk( cudaMalloc(&d_indexes[i],    width * height * sizeof(float)) );
            gpuErrchk( cudaMalloc(&d_lambda[i],     width * height * sizeof(float)) );

            gpuErrchk( cudaMalloc(&d_red_cornerness[i], std::ceil(width /REDUCTION_WINDOW) * std::ceil(height /REDUCTION_WINDOW) * sizeof(float)) );
            gpuErrchk( cudaMalloc(&d_red_indexes[i],    std::ceil(width /REDUCTION_WINDOW) * std::ceil(height /REDUCTION_WINDOW) * sizeof(float)) );

            gpuErrchk( cudaMalloc(&d_red_cornerness_out[i], std::ceil(width /REDUCTION_WINDOW) * std::ceil(height /REDUCTION_WINDOW) * sizeof(float)) );
            gpuErrchk( cudaMalloc(&d_red_indexes_out[i],    std::ceil(width /REDUCTION_WINDOW) * std::ceil(height /REDUCTION_WINDOW) * sizeof(float)) );


            gpuErrchk( cudaMalloc(&d_marker_img[i],  width * height * sizeof(int)) );

            gpuErrchk( cudaMalloc(&d_colours[i], NUM_CORNERS * 4 * sizeof(unsigned int)) );

            gpuErrchk( cudaMalloc(&d_out[i], _channel_size * 3) );
      }
}


// -----------------------------------------------------------------------------
// Function Name : cpyInitDevMem 
//
// Description : copy over device memory that is used throughout rotoscope processing
//			e.g. background image, mask data.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::cpyInitDevMem()
{
      // Copying host input data to d_img and d_back
      if (cv_h_back.isContinuous())
            h_back = (unsigned char*)cv_h_back.datastart;
      gpuErrchk( cudaMemcpy(d_back, h_back, _channel_size * 3, cudaMemcpyHostToDevice) );

      // Copying Mask data to Device
      gpuErrchk( cudaMemcpy(d_gaus_mask,       gaus,   MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(d_gaus_deriv_mask, gaus_d, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice) );
}


// -----------------------------------------------------------------------------
// Function Name : initCUBSorting 
//
// Description : initialize temporary storage space for the cub library Radix Sort.
//			pre initialization of this temporary space reduces the necessity
//			to allocate it each time the RadixSort function is called.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::initCUBSorting()
{

      for (int i = 0; i < NUM_STREAMS; i++)
            d_temp_storage[i] = NULL;

      temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairsDescending(d_temp_storage[0], temp_storage_bytes,
                                                d_red_cornerness[0], d_red_indexes_out[0],
                                                d_red_indexes[0], d_red_indexes[0],
                                                std::ceil(width /REDUCTION_WINDOW) * std::ceil(height /REDUCTION_WINDOW));
      for (int i = 0; i < NUM_STREAMS; i++)
            gpuErrchk( cudaMalloc(&d_temp_storage[i], temp_storage_bytes) );
}


// -----------------------------------------------------------------------------
// Function Name : initCUDAStuff 
//
// Description : initialize cuda stuff, right now only initializing cuda streams.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::initCUDAStuff()
{
      for(int i = 0; i < NUM_STREAMS; i++)
      {
            cudaStreamCreate(&streams[i]);
      }
}


// -----------------------------------------------------------------------------
// Function Name : cpuWatershed 
//
// Description : perform watershed on the CPU using the OpenCV Vincent-Soille 
//			watershed function.
//
// -----------------------------------------------------------------------------

void CUDA_Rotoscope::cpuWatershed()
{
      //************************************************************************
      //        !!!                Perform Watershedding          !!!
      //************************************************************************
      //t_5 = getProcessTime();

      for (int i = 0; i < NUM_STREAMS; i++)
      {
            //------------------------------------------------------------------------
            //                         Copying to Host Memory
            //------------------------------------------------------------------------
            gpuErrchk( cudaMemcpyAsync(h_markers[i], d_marker_img[i], sizeof(int) * width * height, cudaMemcpyDeviceToHost, streams[i]) );
            gpuErrchk( cudaMemcpyAsync(h_diff_img[i], d_diff[i], _channel_size * 3, cudaMemcpyDeviceToHost, streams[i]) );
            //------------------------------------------------------------------------
      }
      cudaDeviceSynchronize();
      cv::Mat cv_marker_img(cv_h_back.size(), CV_32SC1, h_markers[0]);
      cv::Mat cv_diff_img(cv_h_back.size(), CV_8UC3, h_diff_img[0]);




	// ------- VISUAL MARKER DEOMNSTARTION -------
	
      cv::Mat cv_char_marker(cv_h_back.size(), CV_8UC1, cv::Scalar(0));
      cv_marker_img.convertTo(cv_char_marker, CV_8UC1);
      for (int x = 0; x < cv_marker_img.rows; x++)
      {
            for (int y = 0; y < cv_marker_img.cols; y++)
            {
                  if (cv_char_marker.at<uchar>(x,y) < 1)
                        cv_char_marker.at<uchar>(x,y) = 0;
                  else
                        cv_char_marker.at<uchar>(x,y) = 255;
            }
      }
      binaryDilation(cv_char_marker, cv_char_marker);
      binaryDilation(cv_char_marker, cv_char_marker);
      for (int x = 0; x < cv_marker_img.rows; x++)
      {
            for (int y = 0; y < cv_marker_img.cols; y++)
            {
                  if (cv_char_marker.at<uchar>(x,y) == 0)
                        cv_char_marker.at<uchar>(x,y) = 255;
                  else
                        cv_char_marker.at<uchar>(x,y) = 0;
            }
      }
      //cv::namedWindow("Markers",  cv::WINDOW_AUTOSIZE); cv::imshow("Markers",  cv_char_marker);
      //cv::waitKey();
      //cv::namedWindow("Diff",  cv::WINDOW_AUTOSIZE); cv::imshow("Diff",  cv_diff_img);
      //cv::waitKey();

      // cv::Mat cv_corner_img;
      // cv_corner_img = cv::Mat::zeros(cv_h_back.size(),CV_8UC1);
      // for (int i = 0; i < height; i++)
      // {
      //       for (int j = 0; j < width; j++)
      //       {
      //             if (cv_marker_img.at<float>(i,j) !=0)  cv_corner_img.at<uchar>(i,j) = 255;
      //             //std::cout << h_markers[0][i * width + j];
      //       }
      // }
      // cv::namedWindow("Corners",  cv::WINDOW_AUTOSIZE); cv::imshow("Corners",  cv_corner_img);

	// ------------------------------------------


	
	
	// OpenCV Vincent Soille Watershed function call
      cv::watershed(frames[0], cv_marker_img);
      h_markers[0] = (int*)cv_marker_img.datastart;

      //------------------------------------------------------------------------
      //                         Copying to Device Memory
      //------------------------------------------------------------------------
      gpuErrchk( cudaMemcpyAsync(d_marker_img[0], h_markers[0], sizeof(int) * width * height, cudaMemcpyHostToDevice) );
      //------------------------------------------------------------------------

      //t_6 = getProcessTime();
      //************************************************************************
}

// -----------------------------------------------------------------------------
// Function Name : processVideo
//
// Description : heavy lifting function, performs the overall flow of the rotoscope video processing.
//
// -----------------------------------------------------------------------------


void CUDA_Rotoscope::processVideo()
{
      //************************************************************************
      //                         Start of Rotoscoping Frame
      //************************************************************************
      bool process_frame = true;

      dim3 _1Dblock(32);
      dim3 _1Dgrid(std::ceil(NUM_CORNERS / _1Dblock.x));

      //for(int f = 0; f < frame_count; f++)
      for(int f = 0; f < 1; f++)
      {

            //------------------------------------------------------------------------
            //                   Decode next frames to process
            //------------------------------------------------------------------------

            for (int i = 0; i < NUM_STREAMS; i++)
            {
                  //cap.set(cv::CAP_PROP_POS_FRAMES, f);
                  cap.read(frames[i]);
                  if (frames[i].empty())
                  {
                        std::cout << "Could not load frame" << std::endl;
                        process_frame = false;
                  }
                  if (frames[i].isContinuous())
                        h_input[i] = (unsigned char*)frames[i].datastart;
                  else
                        std::cout << "DATA IS NOT CONTINUOUS"<< std::endl;
                  //cv::namedWindow("In",  cv::WINDOW_AUTOSIZE); cv::imshow("In",  frames[i]);
                  //cv::waitKey();
                  //cv::namedWindow("In",  cv::WINDOW_AUTOSIZE); cv::imshow("In",  frames[i]);
            }

            //------------------------------------------------------------------------


            //------------------------------------------------------------------------
            //                   First Round of GPU Functions
            //------------------------------------------------------------------------
		// Calculate block and grid dimensions
		dim3 block(32,16);
            dim3 grid(std::ceil((float)width / block.x), std::ceil((float)height / block.y));
		
		dim3 smooth_sDim = dim3(block.x + MASK_WIDTH - 1, block.y + MASK_WIDTH - 1);
		int  smooth_sz   = smooth_sDim.x * smooth_sDim.y;

		dim3 eigen_sDim = dim3(block.x + CONVOL_WIDTH - 1, block.y + CONVOL_WIDTH);
		int   eigen_sz  = eigen_sDim.x * eigen_sDim.y;

            dim3 window_grid(((width / REDUCTION_WINDOW) / block.x), ((height / REDUCTION_WINDOW) / block.y));

            if (process_frame)
            {
                  for (int i = 0; i < NUM_STREAMS; i++)
                  {


                        //------------------------------------------------------------------------
                        //                         Copying to Device Memory
                        //------------------------------------------------------------------------
                        gpuErrchk( cudaMemcpyAsync(d_img[i],  h_input[i],  _channel_size * 3, cudaMemcpyHostToDevice, streams[i]) );
                        //------------------------------------------------------------------------


                        
                        // Abs difference and convert to grayscale
                        diff_and_convert_kernel<<<grid, block, 0, streams[i]>>>(d_img[i], 
						    							d_back, 
													d_diff[i], 
													d_diff_gray[i], 
													width, height);

                        // ---------------- Feature detection ----------------
                        // Generate Smoothed Images
                        smooth_image_kernel<<<grid, block, smooth_sz * sizeof(unsigned char), streams[i]>>>(d_diff_gray[i], 
						    												d_smooth_x[i], 
																		d_smooth_y[i], 
																		d_gaus_mask, 
																		width, height, 
																		smooth_sDim);

                        // Generate Derivative Images
                        derive_image_kernel<<<grid, block, smooth_sz * sizeof(unsigned char) * 2, streams[i]>>>(d_smooth_x[i], 
						    												d_smooth_y[i], 
																		d_deriv_x[i], 
																		d_deriv_y[i], 
																		d_gaus_deriv_mask, 
																		width, height, 
																		smooth_sDim);

                        // Compute Cornerness Values
                        compute_eigenvalues_kernel<<<grid, block, eigen_sz * sizeof(float) * 2, streams[i]>>>(d_deriv_x[i], 
						    												d_deriv_y[i], 
																		d_cornerness[i], 
																		d_indexes[i], 
																		d_lambda[i], 
																		width, height,
																		eigen_sDim);

                        // Windowing and seperation of high corner values
                        reduceEigenData<<<window_grid, block, 0, streams[i]>>>(d_cornerness[i], 
						    							d_indexes[i], 
													d_red_cornerness[i], 
													d_red_indexes[i], 
													width, height, 
													REDUCTION_WINDOW);


                        cub::DeviceRadixSort::SortPairsDescending(d_temp_storage[i], temp_storage_bytes,
                                                                  d_red_cornerness[i], d_red_cornerness_out[i],
                                                                  d_red_indexes[i], d_red_indexes_out[i],
                                                                  std::ceil(width /REDUCTION_WINDOW) * std::ceil(height /REDUCTION_WINDOW),
                                                                  0, sizeof(float) * 8,
                                                                  streams[i]);

                        clearMarkerImage<<<grid, block, 0, streams[i]>>>(d_marker_img[i], width, height);

                        // Generate Marker Image for watershed
                        generateMarkerImage<<<_1Dgrid, _1Dblock, 0, streams[i]>>>(d_red_indexes_out[i], d_marker_img[i], NUM_CORNERS);
                  }

                  //------------------------------------------------------------------------




                  //************************************************************************
                  //        !!!                Perform Watershedding          !!!
                  //************************************************************************

                  cpuWatershed();

                  //------------------------------------------------------------------------
			



                  //------------------------------------------------------------------------
                  //                   Second Round of GPU Functions
                  //------------------------------------------------------------------------

                  dim3 block(32,16);
                  dim3 grid(cv::divUp(width, block.x), cv::divUp(height, block.y));

                  dim3 pal_block(32);
                  dim3 pal_grid(std::ceil( (4*NUM_CORNERS) / pal_block.x));

                  clear_color_palette<<<pal_block, pal_grid>>>(d_colours[0], NUM_CORNERS);


                  post_water_seg_kernel<<<grid,block>>>(d_img[0],
                                                        d_marker_img[0],
                                                        height, width);


                  color_pal_kernel<<<grid, block>>>(d_img[0],
                                                    d_marker_img[0],
                                                    d_colours[0],
                                                    height, width);

                  avg_color_palette<<<pal_block, pal_grid>>>(d_colours[0], NUM_CORNERS);

                  fill_output_image_kernel<<<grid, block>>>(d_marker_img[0],
                                                            d_colours[0],
                                                            d_out[0],
                                                            height, width);

                  gpuErrchk( cudaMemcpy(h_out[0], d_out[0], _channel_size * 3, cudaMemcpyDeviceToHost) );
                  cudaDeviceSynchronize();
                  //------------------------------------------------------------------------

                  cv::Mat cv_out_img(cv_h_back.size(), CV_8UC3, h_out[0]);
                  output.write(cv_out_img);
                  cv::namedWindow("Out",  cv::WINDOW_AUTOSIZE); cv::imshow("Out",  cv_out_img);
                  cv::waitKey();
            }else{
                  break;
            }
      }
}
