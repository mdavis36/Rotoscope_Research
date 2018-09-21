#include "SerialRotoscope.h"
#include "timer.h"
SerialRotoscope::SerialRotoscope()
{

}

SerialRotoscope::~SerialRotoscope()
{

}

int SerialRotoscope::runRotoscope(int argc, char *argv[])
{
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

      /* OPENCV OPTIMIZATION
      * TODO : Implement gpu::cvtColor
      * https://docs.opencv.org/2.4/modules/gpu/doc/image_processing.html
      */
      // convert to grayscale
      cvtColor( image, image_gray, COLOR_BGR2GRAY );
      cvtColor( image_back, image_back_gray, COLOR_BGR2GRAY );

#ifdef _DISPLAY_ALL
      namedWindow("Original Image", WINDOW_AUTOSIZE ); imshow("Original Image", image);
      namedWindow("Gray Background Image", WINDOW_AUTOSIZE ); imshow("Gray Background Image", image_back_gray);
      namedWindow("Gray Image", WINDOW_AUTOSIZE ); imshow("Gray Image", image_gray);
      waitKey(0);
#endif //_DISPLAY_ALL
      cout << "start frame : " << t_info.start_frame << endl;
      cout << "end frame   : " << t_info.end_frame << endl;

      double t0, t1;
      for( int current_frame = t_info.start_frame; current_frame <= t_info.end_frame; current_frame++){

            if(current_frame > t_info.start_frame){
                  if(!getNextImage(&video, &image, &current_frame)){
                        return -1;
                  }
            }
            t0 = getProcessTime();
            // make difference image

            absdiff( image, image_back, diff_image );


            /*OPENCV OPTIMIZATION TODO : same as previos.*/

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
            /*OPENCV OPTIMIZATION
             * TODO implement openCV class goodFeauturesto track_GPU
             * https://docs.opencv.org/2.4/modules/gpu/doc/video.html
             */
            //t0 = getProcessTime();
            goodFeaturesToTrack(diff_image_gray_ds, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
            //t1 = getProcessTime();


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

            // watershed segmentation

            waterShed_seg(&diff_image, &markers, v_info.ROW, v_info.COL);


            // calculate average color
            out = out.zeros(v_info.ROW,v_info.COL,CV_8UC3); // make output color image


            colorPalette(&image, &markers, &out, color, maxCorners+1, v_info.ROW, v_info.COL); // apply color


            output.write(out);


            //std::cout << "CPU Frame time : " << t1 - t0 << " seconds" << std::endl;
            //namedWindow("Out Image", WINDOW_AUTOSIZE ); imshow("Out Image", out);
            //waitKey(0);
      }




      // display for debugging
//#ifdef _DISPLAY_ALL

//#endif //_DISPLAY_ALL
}


//--------------------------------------------------------------------------------------------------------------------

/* OPENCV OPTIMIZATION
 * TODO : Implement gpu::pyrDown function to leverage GPU accelerated downsampling
 *  https://docs.opencv.org/2.4/modules/gpu/doc/image_processing.html
 */
void SerialRotoscope::downSample(Mat* image, Mat* image_ds, int factor, int COL, int ROW){
	if (factor >= 2) {
		pyrDown(*image, *image_ds, Size(COL/2, ROW/2));
		for(int i  = 2; i < factor; i = i*2){
			pyrDown(*image_ds, *image_ds, Size(COL/2/i, ROW/2/i));
		}
	} else {
		image->copyTo(*image_ds);
	}
}



void SerialRotoscope::GetCenter(vector<Point2f> corners, int* center, int factor){
	Mat center_vector;
	int size  = corners.size();
	/* OPENCV OPTIMIZATION
	 * TODO implement gpu:: reduce
	 * https://docs.opencv.org/2.4/modules/gpu/doc/matrix_reductions.html?highlight=gpu%3A%3Areduce#
	 */
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


/* POSSIBLE OPTIMIZATION TODO
 * OpenMP: For this current build of the code we are only performing 1000 operations
 *         OpenMP parrallelization may be useful.
 *
 *  GPU : If all other openCV optimizations are performed we will have an image of GpuMat
 *  	    Therefore it could be helpful to perform on GPU device with .cu code. Rather than
 *  	    migrating data back to host.
 */
void SerialRotoscope::DrawFeatures_binary(Mat* image, vector<Point2f> corners, int factor) {
	int size  = corners.size();
	for (int i = 0; i < size; ++i) {
		//printf("\nCorner at: x = %d, y = %d", int(corners[i].x * factor), int(corners[i].y * factor));
		image->at<uchar>(Point(int(corners[i].x * factor), int(corners[i].y * factor))) = 255;
	}
}


/* POSSIBLE OPTIMIZATION TODO
 * OpenMP: For this current build of the code we are only performing 1000 operations
 *         OpenMP parrallelization may be useful.
 *
 *  GPU : If all other openCV optimizations are performed we will have an image of GpuMat
 *  	    Therefore it could be helpful to perform on GPU device with .cu code. Rather than
 *  	    migrating data back to host.
 */
void SerialRotoscope::DrawFeatures_markers(Mat* image, vector<Point2f> corners, int factor, int offset) {
	int size  = corners.size();
	for (int i = 0; i < size; ++i) {
		//printf("\nCorner at: x = %d, y = %d", int(corners[i].x * factor), int(corners[i].y * factor));
		image->at<int>(Point(int(corners[i].x * factor), int(corners[i].y * factor))) = i+1+offset;
	}
}



void SerialRotoscope::waterShed_seg(Mat* diff_image, Mat* markers, int ROW, int COL){
	int lab = -1, diff, temp_diff, temp_lab;
	Vec3b val, temp_val;
	watershed(*diff_image, *markers);

	// get rid of boundary pixels
	for(int i = 0; i < ROW; i++){
		for(int j = 0; j < COL; j++){
			// check if pixel is labeled as boundary
			/*
			 *  TODO : profile number of times this if statement is entered compared to size of image
			 *  Possibly a simple OpenMP implementation, otherwise GPU processing should be used to perform
			 *  the convolution.
			 */
			if(markers->at<int>(i,j) == -1){
				diff = 255*3;

				val = diff_image->at<Vec3b>(i,j);

				// check points around pixel
				if(j > 0){
					// upper left
					if(i > 0){
						temp_lab = markers->at<int>(i-1,j-1);
						if(temp_lab > -1){
							temp_val = diff_image->at<Vec3b>(i-1,j-1);
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
						temp_val = diff_image->at<Vec3b>(i,j-1);
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
							temp_val = diff_image->at<Vec3b>(i+1,j-1);
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
						temp_val = diff_image->at<Vec3b>(i-1,j);
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
						temp_val = diff_image->at<Vec3b>(i+1,j);
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
							temp_val = diff_image->at<Vec3b>(i-1,j+1);
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
						temp_val = diff_image->at<Vec3b>(i,j+1);
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
							temp_val = diff_image->at<Vec3b>(i+1,j+1);
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



void SerialRotoscope::colorPalette(Mat* image, Mat* markers, Mat* out, int color[][4], int maxIndex, int ROW, int COL){
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



int SerialRotoscope::init(int argc, char** argv, VideoCapture *video, VideoWriter *output, videoInfo *vi, timeInfo *ti, Mat *image, Mat *image_back)
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

	initTimeInfo(argc, argv, ti, vi);

	// load background and foreground image, and frame counter
	if( !initImages(video, image, image_back, ti->back_time, ti->start_time, vi->FPS, &ti->start_frame) ){
		cout << "Failed to Init Image." << endl;
		return -1;
	}
	return 0;
}
