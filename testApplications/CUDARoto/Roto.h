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

#define MASK_WIDTH 31
#define CONVOL_WIDTH 5
#define K_VAL 0.04


void generateGuassianMasks(int w, float * g, float * g_d);
void diff_and_convert(unsigned char *img,
                      unsigned char *back,
                      unsigned char *o_diff,
                      unsigned char *o_diff_gray,
                      int width,
                      int height,
                      size_t _channel_size);
#endif
