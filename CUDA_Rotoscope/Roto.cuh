#ifndef ROTO_H
#define ROTO_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include <string>


#define MASK_WIDTH 5
#define CONVOL_WIDTH 3
#define REDUCTION_WINDOW 5

#define K_VAL 0.04
#define NUM_STREAMS 1
#define NUM_CORNERS 1000

class CUDA_Rotoscope
{

private:

      //------------------------------------------------------------------------
      //                           Declaring Host Memory
      //------------------------------------------------------------------------
      int *h_markers[NUM_STREAMS];
      unsigned char *h_input[NUM_STREAMS], *h_back, *h_diff_img[NUM_STREAMS], *h_out[NUM_STREAMS];


      //------------------------------------------------------------------------
      //                         Declare Device Memory
      //------------------------------------------------------------------------
      float *d_gaus_mask, *d_gaus_deriv_mask;
      unsigned char *d_back;

      unsigned char *d_img[NUM_STREAMS], *d_diff[NUM_STREAMS], *d_diff_gray[NUM_STREAMS], *d_out[NUM_STREAMS];

      unsigned char *d_smooth_x[NUM_STREAMS];
      unsigned char *d_smooth_y[NUM_STREAMS];
      float *d_deriv_x[NUM_STREAMS];
      float *d_deriv_y[NUM_STREAMS];

      float *d_cornerness[NUM_STREAMS];
      float *d_indexes[NUM_STREAMS];
      float *d_lambda[NUM_STREAMS];
      float *d_red_cornerness[NUM_STREAMS];
      float *d_red_indexes[NUM_STREAMS];

      float *d_red_cornerness_out[NUM_STREAMS];
      float *d_red_indexes_out[NUM_STREAMS];

      int *d_marker_img[NUM_STREAMS];

      unsigned int*d_colours[NUM_STREAMS];
      //------------------------------------------------------------------------


      //------------------------------------------------------------------------
      //                                Video IO
      //------------------------------------------------------------------------
      cv::VideoCapture cap;
      cv::VideoWriter output;

      cv::Mat *frames;
      cv::Mat cv_h_back;

      int height;
      int width;
      int frame_count;
      //------------------------------------------------------------------------


      float *gaus, *gaus_d;
      size_t _channel_size;



      cudaStream_t streams[NUM_STREAMS];
      void *d_temp_storage[NUM_STREAMS];
      size_t temp_storage_bytes;


      void initVideo(std::string filename);
      void generateGaussianMasks();

      void allocHostMem();
      void allocDevMem();
      void cpyInitDevMem();

      void initCUBSorting();
      void initCUDAStuff();

      void cpuWatershed();

public:
      CUDA_Rotoscope(std::string filename);
      ~CUDA_Rotoscope();

      void initialize();

      void processVideo();


};


#endif
