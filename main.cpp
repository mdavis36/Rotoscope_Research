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

#include "videoIO.h"

using namespace videoIO;
using namespace cv;
using namespace std;

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


int init(int argc, char** argv, VideoCapture *video, VideoWriter *output, videoInfo *vi, timeInfo *ti, Mat *image, Mat *image_back);

//rotoscope functions
void downSample(Mat* image, Mat* image_ds, int factor, int COL, int ROW);
void GetCenter(vector<Point2f> corners, int* center, int factor);
void DrawFeatures_binary(Mat* image, vector<Point2f> corners, int factor);
void DrawFeatures_markers(Mat* image, vector<Point2f> corners, int factor, int offset);
void waterShed_seg(Mat* diff_image, Mat* markers, int ROW, int COL);
void colorPalette(Mat* image, Mat* markers, Mat* out, int color[][4], int maxIndex, int ROW, int COL);
//void makeMask(Mat* mask, int* center, int height, int width, int ROW, int COL);

//------------------------------------------------------- MAIN -------------------------------------------------------


int main(int argc, char** argv){


	// Initialization structure variables
	videoInfo v_info;		// Info about input video : FPS, MAX_TIME, MAX_FRAME, ROW, COL
	timeInfo t_info;		// Info about tining to be executed : ,start, end and back time, start/end frame

	int center[2];
	int color[maxCorners+1][4];

	Mat image_back;
	Mat image_back_gray;
	Mat image;
	Mat image_gray;
	Mat diff_image;
	Mat diff_image_gray;
	Mat diff_image_gray_ds;
	Mat corner_image;

	//Mat mask;
	Mat markers;

	Mat out;

	vector<Point2f>	corners;// corners_foreground;
	VideoCapture 	video;
	VideoWriter		output;


	if (init(argc, argv, &video, &output, &v_info, &t_info, &image, &image_back) != 0) return -1;


	// convert to grayscale
	cvtColor( image, image_gray, COLOR_BGR2GRAY );
	cvtColor( image_back, image_back_gray, COLOR_BGR2GRAY );

#ifdef _DISPLAY_ALL
	namedWindow("Original Image", WINDOW_AUTOSIZE ); imshow("Original Image", image);
	namedWindow("Gray Background Image", WINDOW_AUTOSIZE ); imshow("Gray Background Image", image_back_gray);
	namedWindow("Gray Image", WINDOW_AUTOSIZE ); imshow("Gray Image", image_gray);
	waitKey(0);
#endif //_DISPLAY_ALL

	for( int current_frame = t_info.start_frame; current_frame <= t_info.end_frame; current_frame++){

		if(current_frame > t_info.start_frame){
			if(!getNextImage(&video, &image, &current_frame)){
				return -1;
			}
		}

		// make difference image
		absdiff( image, image_back, diff_image );
		cvtColor( diff_image, diff_image_gray, COLOR_BGR2GRAY );

		// downsample image
		downSample(&diff_image_gray, &diff_image_gray_ds, factor, v_info.COL, v_info.ROW);

#ifdef _DISPLAY_ALL
		namedWindow("Diff Image", WINDOW_AUTOSIZE ); imshow("Diff Image", diff_image);
		namedWindow("Diff Gray Image", WINDOW_AUTOSIZE ); imshow("Diff Gray Image", diff_image_gray);
		namedWindow("Diff Gray Image Ds", WINDOW_AUTOSIZE ); imshow("Diff Gray Image Ds", diff_image_gray_ds);
		waitKey(0);
#endif //_DISPLAY_ALL

		// 1st round corner detection
		goodFeaturesToTrack(diff_image_gray_ds, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
		GetCenter(corners, center, factor); // get centroind
		corner_image = corner_image.zeros(v_info.ROW,v_info.COL,CV_8UC1); //make corner grayscale image
		DrawFeatures_binary(&corner_image, corners, factor); // plot corners
		markers = markers.zeros(v_info.ROW,v_info.COL,CV_32SC1); //make markers grayscale image
		DrawFeatures_markers(&markers, corners, factor, 0); // plot markers

#ifdef _DISPLAY_ALL
		namedWindow("Corner Image", WINDOW_AUTOSIZE ); imshow("Corner Image", corner_image);
		namedWindow("Marker Image", WINDOW_AUTOSIZE ); imshow("Marker Image", markers);
		waitKey(0);
#endif //_DISPLAY_ALL

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
		waterShed_seg(&diff_image, &markers, v_info.ROW, v_info.COL);

		// calculate average color
		out = out.zeros(v_info.ROW,v_info.COL,CV_8UC3); // make output color image
		colorPalette(&image, &markers, &out, color, maxCorners+1, v_info.ROW, v_info.COL); // apply color

		output.write(out);
	}

	// display for debugging
#ifdef _DISPLAY_ALL
	namedWindow("Out Image", WINDOW_AUTOSIZE ); imshow("Out Image", out);
	waitKey(0);
#endif //_DISPLAY_ALL

	// exit porgram
	return 0;
}


//--------------------------------------------------------------------------------------------------------------------


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
	reduce(corners, center_vector, 01, REDUCE_AVG);
	Point2f mean(
		center_vector.at<float>(0,0),
		center_vector.at<float>(0,1)
		);

	center[0] = (center_vector.at<float>(0,0)) * factor;
	center[1] = (center_vector.at<float>(0,1)) * factor;

#ifdef _DEBUG
	printf("Number of corners:\t%d\n", size);
	printf("Centroid:\t\t[%d, %d]\n\n", center[0], center[1]);
#endif //_DEBUG
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

/*
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
*/

int init(int argc, char** argv, VideoCapture *video, VideoWriter *output, videoInfo *vi, timeInfo *ti, Mat *image, Mat *image_back)
{
	// check input arg
	if ( argc < 3  || argc > 5) {
		printf("usage: Program <Image_Path> <Start_Time>\n");
		printf("optional usages:\n");
		printf("\tProgram <Image_Path> <Start_Time> <End_Time>\n");
		printf("\tProgram <Image_Path> <Start_Time> <End_Time> <Background_Time>\n");
		return -1; //exit with error
	}

	//load video and validate user input
	if( !loadVideo(argv[1], video) ){ // open video file
		cout << "Failed to load input video.\n";
		return -1; //exit with error
	}

	getVideoProperties(video, vi, &ti->start_time); // get local properties for video file

	if( !initVideoOutput(argv[1], output, vi) ){ // open video file
		cout << "Failed to initailize output video.\n";
		return -1; //exit with error
	}

	if( !checkStartTime(argv[2], &ti->start_time, vi->MAX_TIME) ){ // check user input for start time
		return -1; //exit with error
	}

	if(argc > 3){
		if( !checkEndTime(argv[3], &ti->start_time, &ti->end_time, vi->MAX_TIME) ){ // check user input for start time
			return -1; //exit with error
		}
	} else {
		ti->end_time = ti->start_time; //only make 1 frame
	}

	if(argc > 4){
		if( !checkStartTime(argv[4], &ti->back_time, vi->MAX_TIME) ){ // check user input for start time
			return -1; //exit with error
		}
	} else {
		ti->back_time = 0; //default to first frame
	}
	ti->end_frame = ti->end_time * vi->FPS;

	// load background and foreground image, and frame counter
	if( !initImages(video, image, image_back, ti->back_time, ti->start_time, vi->FPS, &ti->start_frame) ){
		cout << "Failed to Init Image." << endl;
		return -1;
	}

	return 0;
}
