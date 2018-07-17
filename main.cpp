/***********************************************************
**| Digital Rotoscoping:				 |**
**| 	Implementation 1 in C++ to match MATLAB code for |**
**|	digital Rotoscoping.				 |**
**|							 |**
**|	By: Iain Murphy					 |**
**|							 |**
***********************************************************/

// Example Usage
//	./main images/Louis.mp4 16
//	./main images/Louis.mp4 16 18
//	./main images/Louis.mp4 16 18 0


#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;


// debug flags
	int display_all = 0;
	int debug = 0;


// globals
	// for downsampleing
	int factor = 2;
	// for corner detection
	int maxCorners 		= 1000;
	double qualityLevel 	= 0.000001;
	double minDistance 	= 1;
	int blockSize 		= 3;
	bool useHarrisDetector 	= false;
	double k 		= 0.04;

	//int maxTrackbar = 100;
	//output video codec
	int codec = CV_FOURCC('M','J','P','G');



//initializing video
int loadVideo(char* filename, VideoCapture* video);
void getVideoProperties(int debug, VideoCapture* video, double* FPS, double* MAX_TIME, double* start_time, int* MAX_FRAME, int* ROW, int* COL);
int initVideoOutput(char* filename, VideoWriter* output, int* ROW, int* COL, double* FPS);

//validating user input
int checkStartTime(int debug, char* time, double* start_time, double MAX_TIME);
int checkEndTime(int debug, char* time, double* start_time, double* end_time, double MAX_TIME);

//reading video
int initImages(VideoCapture* video, Mat* image, Mat* image_back, double back_time, double start_time, double FPS, int* current_frame);
int getNextImage(int debug, VideoCapture* video, Mat* image, int* current_frame);

//rotoscope functions
void downSample(Mat* image, Mat* image_ds, int factor, int COL, int ROW);
void GetCenter(vector<Point2f> corners, int* center, int factor);
void DrawFeatures_binary(Mat* image, vector<Point2f> corners, int factor);
void DrawFeatures_markers(Mat* image, vector<Point2f> corners, int factor, int offset);
void makeMask(Mat* mask, int* center, int height, int width, int ROW, int COL);
void waterShed_seg(Mat* diff_image, Mat* markers, int ROW, int COL);
void colorPalette(Mat* image, Mat* markers, Mat* out, int color[][4], int maxIndex, int ROW, int COL);



int main(int argc, char** argv){

	double 	FPS, MAX_TIME, start_time,
		end_time, back_time;
	int 	MAX_FRAME, ROW, COL,
		current_frame, center[2],color[maxCorners+1][4],
		end_frame, start_frame;
	Mat 	image_back, image_back_gray, image,
		image_gray, diff_image, diff_image_gray,
		diff_image_gray_ds,corner_image, mask,
		markers, out;
	vector<Point2f>	corners, corners_foreground;
	VideoCapture 	video;
	VideoWriter	output;


	// check input arg
	if ( argc < 3  || argc > 5) {
		printf("usage: Program <Image_Path> <Start_Time>\n");
		printf("optional usages:\n");
		printf("\tProgram <Image_Path> <Start_Time> <End_Time>\n");
		printf("\tProgram <Image_Path> <Start_Time> <End_Time> <Background_Time>\n");
		return -1; //exit with error
	}


	//load video and validate user input
	if( !loadVideo(argv[1], &video) ){ // open video file
		return -1; //exit with error
	}

	getVideoProperties(debug, &video, &FPS, &MAX_TIME, &start_time, &MAX_FRAME, &ROW, &COL); // get local properties for video file

	if( !initVideoOutput(argv[1], &output, &ROW, &COL, &FPS) ){ // open video file
		return -1; //exit with error
	}

	if( !checkStartTime(debug, argv[2], &start_time, MAX_TIME) ){ // check user input for start time
		return -1; //exit with error
	}

	if(argc > 3){
		if( !checkEndTime(debug, argv[3], &start_time, &end_time, MAX_TIME) ){ // check user input for start time
			return -1; //exit with error
		}
	} else {
		end_time = start_time; //only make 1 frame
	}
	end_frame = end_time*FPS;

	if(argc > 4){
		if( !checkStartTime(debug, argv[4], &back_time, MAX_TIME) ){ // check user input for start time
			return -1; //exit with error
		}
	} else {
		back_time = 0; //default to first frame
	}

	// load background and foreground image, and frame counter
	if( !initImages(&video, &image, &image_back, back_time, start_time, FPS, &start_frame) ){
		return -1;
	}

	// convert to grayscale
	cvtColor( image, image_gray, CV_BGR2GRAY );
	cvtColor( image_back, image_back_gray, CV_BGR2GRAY );

	if( display_all ){
		namedWindow("Original Image", WINDOW_AUTOSIZE ); imshow("Original Image", image);
		namedWindow("Gray Background Image", WINDOW_AUTOSIZE ); imshow("Gray Background Image", image_back_gray);
		namedWindow("Gray Image", WINDOW_AUTOSIZE ); imshow("Gray Image", image_gray);
		waitKey(0);
	}

	for(current_frame = start_frame; current_frame <= end_frame; current_frame++){

		if(current_frame > start_frame){
			if(!getNextImage(debug, &video, &image, &current_frame)){
				return -1;
			}
		}
	
		// make difference image
		absdiff( image, image_back, diff_image );
		cvtColor( diff_image, diff_image_gray, CV_BGR2GRAY );

		// downsample image
		downSample(&diff_image_gray, &diff_image_gray_ds, factor, COL, ROW);

		if( display_all ){
			namedWindow("Diff Image", WINDOW_AUTOSIZE ); imshow("Diff Image", diff_image);
			namedWindow("Diff Gray Image", WINDOW_AUTOSIZE ); imshow("Diff Gray Image", diff_image_gray);
			namedWindow("Diff Gray Image Ds", WINDOW_AUTOSIZE ); imshow("Diff Gray Image Ds", diff_image_gray_ds);
			waitKey(0);
		}

		// 1st round corner detection
		goodFeaturesToTrack(diff_image_gray_ds, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
		GetCenter(corners, center, factor); // get centroind
		corner_image = corner_image.zeros(ROW,COL,CV_8UC1); //make corner grayscale image
		DrawFeatures_binary(&corner_image, corners, factor); // plot corners
		markers = markers.zeros(ROW,COL,CV_32SC1); //make markers grayscale image
		DrawFeatures_markers(&markers, corners, factor, 0); // plot markers

		if( display_all ){
			namedWindow("Corner Image", WINDOW_AUTOSIZE ); imshow("Corner Image", corner_image);
			namedWindow("Marker Image", WINDOW_AUTOSIZE ); imshow("Marker Image", markers);
			waitKey(0);
		}

		/*
		// 2nd round corner detection
		mask = mask.zeros(ROW,COL,CV_8UC1);
		makeMask(&mask, center, 120, 120, ROW, COL);
		goodFeaturesToTrack(diff_image_gray, corners_foreground, maxCorners/2, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
		if( display_all ){
			namedWindow("Mask Image", WINDOW_AUTOSIZE ); imshow("Mask Image", mask);
			waitKey(0);
		}

		if ( debug ) {
			int size = corners_foreground.size();
			printf("Number of corners:\t%d\n", size );
		}
		*/

		// watershed segmentation
		waterShed_seg(&diff_image, &markers, ROW, COL);

		// calculate average color
		out = out.zeros(ROW,COL,CV_8UC3); // make output color image
		colorPalette(&image, &markers, &out, color, maxCorners+1, ROW, COL); // apply color

		output.write(out);

	}
	

	// display for debugging
	if( display_all ){
		namedWindow("Out Image", WINDOW_AUTOSIZE ); imshow("Out Image", out);
		waitKey(0);
	}

	// exit porgram
	return 0;
}




int loadVideo(char* filename, VideoCapture* video){
	video->open(filename);
	if (!video->isOpened()) {
		printf("No video data \n");
		return 0;
	}
	return 1;
}



void getVideoProperties(int debug, VideoCapture* video, double* FPS, double* MAX_TIME, double* start_time, int* MAX_FRAME, int* ROW, int* COL){
	*FPS = video->get(CV_CAP_PROP_FPS);
	*MAX_FRAME = video->get(CV_CAP_PROP_FRAME_COUNT);
	*ROW = video->get(CV_CAP_PROP_FRAME_HEIGHT);
	*COL = video->get(CV_CAP_PROP_FRAME_WIDTH);
	*MAX_TIME = ((double)(*MAX_FRAME))/(*FPS);

	if(debug){
		printf("\nVideo Properties:\n");
		printf("\tFPS = \t\t%g fps\n", *FPS);
		printf("\tMax Frame = \t%d frames\n", *MAX_FRAME);
		printf("\tMax Time = \t%g sec\n", *MAX_TIME);
		printf("\tHeight = \t%d pixels\n", *ROW);
		printf("\tWidth = \t%d pixels\n", *COL);
	}
}



int initVideoOutput(char* filename, VideoWriter* output, int* ROW, int* COL, double* FPS){
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
	printf("\nOutput Video = %s\n", name);

	output->open(name, codec, *FPS, (Size) Size(*COL,*ROW), true);

	if(!output->isOpened()){
		printf("Failed to open Output Video\n"); 
		return 0;
	}

	/*printf("\nOutput Video Properties:\n");
	printf("\tHeight = \t%d pixels\n", (int)output->get(CV_CAP_PROP_FRAME_HEIGHT));
	printf("\tWidth = \t%d pixels\n", (int)output->get(CV_CAP_PROP_FRAME_WIDTH));*/
	return 1;
	
	
}


int checkStartTime(int debug, char* time, double* start_time, double MAX_TIME){
	char* err;

	*start_time = strtod(time, &err);
	if(time == err || *err != 0 || *start_time > MAX_TIME || *start_time < 0) {
		printf("\nInvalid Start Time: %s\n", time);
		return 0;
	}
	if(debug){
		printf("\tStart Time = \t%g sec\n\n", *start_time);
	}
	return 1;
}



int checkEndTime(int debug, char* time, double* start_time, double* end_time, double MAX_TIME){
	char* err;

	*end_time = strtod(time, &err);
	if(time == err || *err != 0 || *end_time > MAX_TIME || *end_time < 0 || *end_time < *start_time) {
		printf("\nInvalid End Time: %s\n", time);
		return 0;
	}
	if(debug){
		printf("\tEnd Time = \t%g sec\n\n", *start_time);
	}
	return 1;
}



int initImages(VideoCapture* video, Mat* image, Mat* image_back, double back_time, double start_time, double FPS, int* current_frame){
	if(back_time > 0){
		*current_frame = back_time*FPS;
		video->set(CV_CAP_PROP_POS_FRAMES, *current_frame);
	}
	video->read(*image_back);
	if ( !image_back->data ) {
		printf("No background image data \n");
		return 0;
	}

	*current_frame = start_time*FPS;
	video->set(CV_CAP_PROP_POS_FRAMES, *current_frame);
	video->read(*image);
	if ( !image->data ) {
		printf("No image data \n");
		return 0;
	}
	return 1;
}



int getNextImage(int debug, VideoCapture* video, Mat* image, int* current_frame){
	if(debug){
		printf("Getting next frame: Frame Number = %d\n", *current_frame);
	}
	video->set(CV_CAP_PROP_POS_FRAMES, *current_frame);
	video->read(*image);
	if ( !image->data ) {
		printf("No image data \n");
		return 0;
	}
	return 1;
}


void downSample(Mat* image, Mat* image_ds, int factor, int COL, int ROW){
	if (factor >= 2) {
		pyrDown(*image, *image_ds, Size(COL/2, ROW/2));
		for(int i  = 2; i < factor; i = i*2){
			pyrDown(*image_ds, *image_ds, Size(COL/2/i, ROW/2/i));
		}
	} else {
		image->copyTo(*image_ds);
	}
}



void GetCenter(vector<Point2f> corners, int* center, int factor){
	Mat center_vector;
	int size  = corners.size();
	reduce(corners, center_vector, 01, CV_REDUCE_AVG);
	Point2f mean(
		center_vector.at<float>(0,0),
		center_vector.at<float>(0,1)
		);

	center[0] = (center_vector.at<float>(0,0)) * factor; 
	center[1] = (center_vector.at<float>(0,1)) * factor;

	if ( debug ) {
		printf("Number of corners:\t%d\n", size);
		printf("Centroid:\t\t[%d, %d]\n\n", center[0], center[1]);
	}
}



void DrawFeatures_binary(Mat* image, vector<Point2f> corners, int factor) {
	int size  = corners.size();
	for (int i = 0; i < size; ++i) {
		//printf("\nCorner at: x = %d, y = %d", int(corners[i].x * factor), int(corners[i].y * factor));
		image->at<uchar>(Point(int(corners[i].x * factor), int(corners[i].y * factor))) = 255;
	}
}



void DrawFeatures_markers(Mat* image, vector<Point2f> corners, int factor, int offset) {
	int size  = corners.size();
	for (int i = 0; i < size; ++i) {
		//printf("\nCorner at: x = %d, y = %d", int(corners[i].x * factor), int(corners[i].y * factor));
		image->at<int>(Point(int(corners[i].x * factor), int(corners[i].y * factor))) = i+1+offset;
	}
}



void makeMask(Mat* mask, int* center, int height, int width, int ROW, int COL){
	int limits[4] = {0, ROW, 0, COL};
	if(center[0]-height > 0){
		limits[0] = center[0]-height;
	}
	if(center[0]+height < ROW){
		limits[1] = center[0]-height;
	}
	if(center[1]-width > 0){
		limits[2] = center[1]-width;
	}
	if(center[1]+width < ROW){
		limits[3] = center[1]+width;
	}
	for(int i = limits[0]; i < limits[1]; i++){
		for(int j = limits[2]; j < limits[3]; j++){
			mask->at<char>(i,j) = (char)255;
		}
	}
}



void waterShed_seg(Mat* diff_image, Mat* markers, int ROW, int COL){
	int lab = -1, diff, val[3], temp_val[3], temp_diff, temp_lab;
	watershed(*diff_image, *markers);
	// get rid of boundary pixels
	for(int i = 0; i < ROW; i++){
		for(int j = 0; j < COL; j++){
			// check if pixel is labeled as boundary 
			if(markers->at<int>(i,j) == -1){
				diff = 255*3;

				val[0] = diff_image->at<Vec3b>(i,j)[0];
				val[1] = diff_image->at<Vec3b>(i,j)[1];
				val[2] = diff_image->at<Vec3b>(i,j)[2];
				
				// check points around pixel
				if(j > 0){
					// upper left
					if(i > 0){
						temp_lab = markers->at<int>(i-1,j-1);
						if(temp_lab > -1){
							temp_val[0] = diff_image->at<Vec3b>(i-1,j-1)[0];
							temp_val[1] = diff_image->at<Vec3b>(i-1,j-1)[1];
							temp_val[2] = diff_image->at<Vec3b>(i-1,j-1)[2];
							temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
							if (temp_diff < diff){
								diff = temp_diff;
								lab = temp_lab;
							}
						}
					}
					// above
					temp_lab = markers->at<int>(i,j-1);
					if(temp_lab > -1){
						temp_val[0] = diff_image->at<Vec3b>(i,j-1)[0];
						temp_val[1] = diff_image->at<Vec3b>(i,j-1)[1];
						temp_val[2] = diff_image->at<Vec3b>(i,j-1)[2];
						temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
						if (temp_diff < diff){
							diff = temp_diff;
							lab = temp_lab;
						}
					}
					// upper right
					if(i < ROW-1){
						temp_lab = markers->at<int>(i+1,j-1);
						if(temp_lab > -1){
							temp_val[0] = diff_image->at<Vec3b>(i+1,j-1)[0];
							temp_val[1] = diff_image->at<Vec3b>(i+1,j-1)[1];
							temp_val[2] = diff_image->at<Vec3b>(i+1,j-1)[2];
							temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
							if (temp_diff < diff){
								diff = temp_diff;
								lab = temp_lab;
							}
						}
					}
				}
				// left
				if(i > 0){
					temp_lab = markers->at<int>(i-1,j);
					if(temp_lab > -1){
						temp_val[0] = diff_image->at<Vec3b>(i-1,j)[0];
						temp_val[1] = diff_image->at<Vec3b>(i-1,j)[1];
						temp_val[2] = diff_image->at<Vec3b>(i-1,j)[2];
						temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
						if (temp_diff < diff){
							diff = temp_diff;
							lab = temp_lab;
						}
					}
				}
				// right
				if(i < ROW-1){
					temp_lab = markers->at<int>(i+1,j);
					if(temp_lab > -1){
						temp_val[0] = diff_image->at<Vec3b>(i+1,j)[0];
						temp_val[1] = diff_image->at<Vec3b>(i+1,j)[1];
						temp_val[2] = diff_image->at<Vec3b>(i+1,j)[2];
						temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
						temp_lab = markers->at<int>(i+1,j);
						if (temp_diff < diff){
							diff = temp_diff;
							lab = temp_lab;
						}
					}
				}
				if(j < COL-1){
					// bottom left
					if(i > 0){
						temp_lab = markers->at<int>(i-1,j+1);
						if(temp_lab > -1){
							temp_val[0] = diff_image->at<Vec3b>(i-1,j+1)[0];
							temp_val[1] = diff_image->at<Vec3b>(i-1,j+1)[1];
							temp_val[2] = diff_image->at<Vec3b>(i-1,j+1)[2];
							temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
							if (temp_diff < diff && temp_lab > -1){
								diff = temp_diff;
								lab = temp_lab;
							}
						}
					}
					// below
					temp_lab = markers->at<int>(i,j+1);
					if(temp_lab > -1){
						temp_val[0] = diff_image->at<Vec3b>(i,j+1)[0];
						temp_val[1] = diff_image->at<Vec3b>(i,j+1)[1];
						temp_val[2] = diff_image->at<Vec3b>(i,j+1)[2];
						temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
						if (temp_diff < diff){
							diff = temp_diff;
							lab = temp_lab;
						}
					}
					// bottom right
					if(i < ROW-1){
						temp_lab = markers->at<int>(i+1,j+1);
						if(temp_lab > -1){
							temp_val[0] = diff_image->at<Vec3b>(i+1,j+1)[0];
							temp_val[1] = diff_image->at<Vec3b>(i+1,j+1)[1];
							temp_val[2] = diff_image->at<Vec3b>(i+1,j+1)[2];
							temp_diff = abs(val[0] - temp_val[0]) + abs(val[1] - temp_val[1]) + abs(val[2] - temp_val[2]);
							if (temp_diff < diff && temp_lab > -1){
								diff = temp_diff;
								lab = temp_lab;
							}
						}
					}
				}
				// assign new label
				markers->at<int>(i,j) = lab;
			}
		}
	}
}



void colorPalette(Mat* image, Mat* markers, Mat* out, int color[][4], int maxIndex, int ROW, int COL){
	int i,j,index;
	for(i = 0; i < maxIndex; i++){
		color[i][0] = 0;
		color[i][1] = 0;
		color[i][2] = 0;
		color[i][3] = 0;
	}
	for(i = 0; i < ROW; i++ ){
                for(j = 0; j < COL; j++ ){
			index = markers->at<int>(i,j);
			if (index > -1){
				color[index][3] = color[index][3] + 1;
				color[index][0] = color[index][0] + int(image->at<Vec3b>(i,j)[0]);
				color[index][1] = color[index][1] + int(image->at<Vec3b>(i,j)[1]);
				color[index][2] = color[index][2] + int(image->at<Vec3b>(i,j)[2]);
			}
		}
	}
	for(i = 0; i < maxIndex; i++){
		index = color[i][3];
		color[i][0] = color[i][0]/index;
		color[i][1] = color[i][1]/index;
		color[i][2] = color[i][2]/index;
	}
	for(i = 0; i < ROW; i++ ){
                for(j = 0; j < COL; j++ ){
			index = markers->at<int>(i,j);
			if (index > -1){
				out->at<Vec3b>(i,j)[0] = color[index][0];
				out->at<Vec3b>(i,j)[1] = color[index][1];
				out->at<Vec3b>(i,j)[2] = color[index][2];
			}
		}
	}
}



