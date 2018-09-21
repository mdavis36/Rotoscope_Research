#ifndef Roto_H
#define Roto_H

#include <iostream>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace std;
//#include "timer.h"

#define MASK_WIDTH 5
#define CONVOL_WIDTH 3
#define K_VAL 0.04


void generateGuassianMasks(int w, float * g, float * g_d);
void rotoscope(unsigned char *img,
               unsigned char *back,
               unsigned char *o_diff,
               unsigned char *o_diff_gray,
               unsigned char *h_corners,
               unsigned char *h_smooth_x,
               unsigned char *h_smooth_y,
               unsigned char *h_deriv_x,
               unsigned char *h_deriv_y,
               int width,
               int height,
               size_t _channel_size);
#endif
