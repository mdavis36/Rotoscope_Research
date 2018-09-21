#ifndef SERIALROTOSCOPE_H
#define SERIALROTOSCOPE_H

#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "videoIO.h"


using namespace videoIO;
using namespace cv;
using namespace std;


class SerialRotoscope
{
public:
      int factor              = 2;
      int maxCorners 		= 1000;
      double qualityLevel 	= 0.000001;
      double minDistance 	= 1;
      int blockSize 		= 3;
      bool useHarrisDetector 	= false;
      double k 		      = 0.04;

      SerialRotoscope();
      ~SerialRotoscope();

      int runRotoscope(int argc, char* argv[]);

private:
      int init(int argc, char** argv, VideoCapture *video, VideoWriter *output, videoInfo *vi, timeInfo *ti, Mat *image, Mat *image_back);
      void downSample(Mat* image, Mat* image_ds, int factor, int COL, int ROW);
      void GetCenter(vector<Point2f> corners, int* center, int factor);
      void DrawFeatures_binary(Mat* image, vector<Point2f> corners, int factor);
      void DrawFeatures_markers(Mat* image, vector<Point2f> corners, int factor, int offset);
      void waterShed_seg(Mat* diff_image, Mat* markers, int ROW, int COL);
      void colorPalette(Mat* image, Mat* markers, Mat* out, int color[][4], int maxIndex, int ROW, int COL);

};

#endif
