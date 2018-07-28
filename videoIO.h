#ifndef VIDEOIO_H
#define VIDEOIO_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;

//int codec = VideoWriter::fourcc('M','J','P','G');

namespace videoIO
{

      typedef struct
      {
            double FPS;
      	double MAX_TIME;
      	int MAX_FRAME;
      	int ROW;
      	int COL;
      } videoInfo;

      typedef struct
      {
      	double start_time;
      	double end_time;
      	double back_time;
      	int start_frame;
      	int end_frame;
      } timeInfo ;

      //handleTimeInfo
      int initTimeInfo(int argc, char** argv, timeInfo *ti, videoInfo *vi);

      //initializing video
      int loadVideo(char* filename, VideoCapture* video);
      void getVideoProperties(VideoCapture* video, videoInfo *vi, double* start_time);
      int initVideoOutput(char* filename, VideoWriter* output, videoInfo *vi);

      //reading video
      int initImages(VideoCapture* video, Mat* image, Mat* image_back, double back_time, double start_time, double FPS, int* current_frame);
      int getNextImage(VideoCapture* video, Mat* image, int* current_frame);

      //validating user input
      int checkStartTime(char* time, double* start_time, double MAX_TIME);
      int checkEndTime(char* time, double* start_time, double* end_time, double MAX_TIME);
}

#endif
