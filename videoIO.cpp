#include "videoIO.h"

namespace videoIO
{

      int loadVideo(char* filename, VideoCapture* video){
      	video->open(filename);
      	if (!video->isOpened()) {
      		printf("No video data \n");
      		return 0;
      	}
      	return 1;
      }



      void getVideoProperties(VideoCapture* video, videoInfo *vi, double* start_time){
      	vi->FPS = video->get(CAP_PROP_FPS);
      	vi->MAX_FRAME = video->get(CAP_PROP_FRAME_COUNT);
      	vi->ROW = video->get(CAP_PROP_FRAME_HEIGHT);
      	vi->COL = video->get(CAP_PROP_FRAME_WIDTH);
      	vi->MAX_TIME = ((double)(vi->MAX_FRAME))/(vi->FPS);

      #ifdef _DEBUG
      	printf("\nVideo Properties:\n");
      	printf("\tFPS = \t\t%g fps\n", *FPS);
      	printf("\tMax Frame = \t%d frames\n", *MAX_FRAME);
      	printf("\tMax Time = \t%g sec\n", *MAX_TIME);
      	printf("\tHeight = \t%d pixels\n", *ROW);
      	printf("\tWidth = \t%d pixels\n", *COL);
      #endif //_DEBUG
      }



      int initVideoOutput(char* filename, VideoWriter* output, videoInfo *vi){
      	char name[256];
      	char* extPtr;
      	char* temp;

      	strcpy(name, filename);
      	temp = strchr(name, '.');
      	while(temp != NULL){
      		extPtr = temp;
      		temp = strchr(extPtr+1, '.');
      	}
      	extPtr[0] = '\0';
      	extPtr++;
      	strcat(name, "_roto.avi");
      	printf("Output Video = %s\n", name);

            int codec = VideoWriter::fourcc('M','J','P','G');
      	output->open(name, codec, vi->FPS, (Size) Size(vi->COL,vi->ROW), true);

      	if(!output->isOpened()){
      		printf("Failed to open Output Video\n");
      		return 0;
      	}

      	/*printf("\nOutput Video Properties:\n");
      	printf("\tHeight = \t%d pixels\n", (int)output->get(CV_CAP_PROP_FRAME_HEIGHT));
      	printf("\tWidth = \t%d pixels\n", (int)output->get(CV_CAP_PROP_FRAME_WIDTH));*/
      	return 1;
      }


      int initImages(VideoCapture* video, Mat* image, Mat* image_back, double back_time, double start_time, double FPS, int* current_frame){
      	if(back_time > 0){
      		*current_frame = back_time*FPS;
      		video->set(CAP_PROP_POS_FRAMES, *current_frame);
      	}
      	video->read(*image_back);
      	if ( !image_back->data ) {
      		printf("No background image data \n");
      		return 0;
      	}

      	*current_frame = start_time*FPS;
      	video->set(CAP_PROP_POS_FRAMES, *current_frame);
      	video->read(*image);
      	if ( !image->data ) {
      		printf("No image data \n");
      		return 0;
      	}
      	return 1;
      }



      int getNextImage(VideoCapture* video, Mat* image, int* current_frame){

      #ifdef _DEBUG
      	printf("Getting next frame: Frame Number = %d\n", *current_frame);
      #endif //_DEBUG

      	video->set(CAP_PROP_POS_FRAMES, *current_frame);
      	video->read(*image);
      	if ( !image->data ) {
      		printf("No image data \n");
      		return 0;
      	}
      	return 1;
      }



      int checkStartTime(char* time, double* start_time, double MAX_TIME){
      	char* err;

      	*start_time = strtod(time, &err);
      	if(time == err || *err != 0 || *start_time > MAX_TIME || *start_time < 0) {
      		printf("\nInvalid Start Time: %s\n", time);
      		return 0;
      	}

      #ifdef _DEBUG
      	printf("\tStart Time = \t%g sec\n\n", *start_time);
      #endif //_DEBUG

      }



      int checkEndTime(char* time, double* start_time, double* end_time, double MAX_TIME){
      	char* err;

      	*end_time = strtod(time, &err);
      	if(time == err || *err != 0 || *end_time > MAX_TIME || *end_time < 0 || *end_time < *start_time) {
      		printf("\nInvalid End Time: %s\n", time);
      		return 0;
      	}
      #ifdef _DEBUG
      	printf("\tEnd Time = \t%g sec\n\n", *start_time);
      #endif //_DEBUG
      	return 1;
      }


}
