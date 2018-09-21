#include <opencv2/opencv.hpp>

#include "Roto.h"

void show3b(string title, uchar *data, int rows, int cols)
{
      cv::Mat out = cv::Mat(rows, cols, CV_8UC3);
      int indx = 0;
      for (int i = 0; i < rows; i++)
      {
            for (int j = 0; j < cols; j++)
            {
                  out.at<cv::Vec3b>(i,j) = cv::Vec3b(data[indx],
                                                     data[indx+1],
                                                     data[indx+2]);
                  indx += 3;
            }
      }
      cv::namedWindow(title, cv::WINDOW_AUTOSIZE ); cv::imshow(title, out);
}

void showuchar(string title, uchar *data, int rows, int cols)
{
      cv::Mat out = cv::Mat(rows, cols, CV_8UC1);
      int indx = 0;
      for (int i = 0; i < rows; i++)
      {
            for (int j = 0; j < cols; j++)
            {
                  out.at<uchar>(i,j) = data[indx];
                  indx ++;
            }
      }
      cv::namedWindow(title, cv::WINDOW_AUTOSIZE ); cv::imshow(title, out);
}

int main()
{
      int num_rows, num_cols;

      // Read test input images
      cv::Mat h_image = cv::imread("img.png",cv::IMREAD_COLOR);
      cv::Mat h_back = cv::imread("back.png",cv::IMREAD_COLOR);

      // Vectors to store data from OpenCV Mat datatypes
      std::vector<unsigned char> v_image, v_back;

      // Convert to 1D contiguous vecors
      if (h_image.isContinuous())
            v_image.assign(h_image.datastart, h_image.dataend);

      if (h_back.isContinuous())
            v_back.assign(h_back.datastart, h_back.dataend);

      // Create and allocate memory for output arrays
      unsigned char *diff, *diff_gray, *corners, *red_corners;
      size_t _channel_size = sizeof(unsigned char) * h_image.cols * h_image.rows;
      diff = (unsigned char *)malloc(_channel_size * 3);
      diff_gray = (unsigned char *)malloc(_channel_size);
      corners = (unsigned char *)malloc(_channel_size);

      unsigned char *smooth_x;
      unsigned char *smooth_y;
      unsigned char *deriv_x;
      unsigned char *deriv_y;

      smooth_x = (unsigned char *)malloc(_channel_size);
      smooth_y = (unsigned char *)malloc(_channel_size);
      deriv_x  = (unsigned char *)malloc(_channel_size);
      deriv_y  = (unsigned char *)malloc(_channel_size);

      // First CUDA Kernel call function performs both absdiff and cvtColor in one.
      rotoscope(v_image.data(),
                v_back.data(),
                diff,
                diff_gray,
                corners,
                smooth_x,
                smooth_y,
                deriv_x,
                deriv_y,
                h_image.cols,
                h_image.rows,
                _channel_size);

      // -----------------------------------------------------------------------
      //
      // Purely to be used to check the output of the data from the GPU kernels
      //
      // -----------------------------------------------------------------------

      show3b("GPU Image", v_image.data(), h_image.rows, h_image.cols);
      show3b("GPU Difference", diff, h_image.rows, h_image.cols);
      showuchar("GPU Diff Gray", diff_gray, h_image.rows, h_image.cols);
      showuchar("GPU Smooth X", smooth_x, h_image.rows, h_image.cols);
      showuchar("GPU Smooth Y", smooth_y, h_image.rows, h_image.cols);
      showuchar("GPU Derive X", deriv_x,  h_image.rows, h_image.cols);
      showuchar("GPU Derive Y", deriv_y,  h_image.rows, h_image.cols);
      showuchar("GPU Corners", corners, h_image.rows, h_image.cols);




      // -----------------------------------------------------------------------
      //                            CPU IMPLEMENTATION
      // -----------------------------------------------------------------------

      int maxCorners 		= 1000;
      double qualityLevel 	= 0.000001;
      double minDistance 	= 1;
      int blockSize 		= 3;
      bool useHarrisDetector 	= false;
      double k 		      = 0.04;
      int factor              = 1;

      cv::Mat h_diff_image, h_diff_image_gray, h_out;
      cv::Mat h_corners;

      std::vector<cv::Point2f> cpu_corners;

      cv::absdiff(h_image, h_back, h_diff_image);
      cv::cvtColor(h_diff_image, h_diff_image_gray, cv::COLOR_BGR2GRAY);

      // - DownSampling
      //
      // if (factor >=2)
      // {
      //       cv::pyrDown(h_diff_image_gray, h_diff_image_gray);
      //       for(int i = 2; i < factor; i = i*2)
      //       {
      //             cv::pyrDown(h_diff_image_gray, h_diff_image_gray);
      //       }
      // }

      cv::goodFeaturesToTrack(h_diff_image_gray, cpu_corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

      cv::Mat marker_image = cv::Mat::zeros(h_image.size(), CV_32SC1);
      cv::Mat corner_image = cv::Mat::zeros(h_image.size(), CV_8UC1);

      for (int i = 0; i < cpu_corners.size(); i++)
      {
            marker_image.at<int>(cv::Point(int(cpu_corners[i].x * factor), int(cpu_corners[i].y * factor))) = i+1;
            corner_image.at<uchar>(cv::Point(int(cpu_corners[i].x * factor), int(cpu_corners[i].y * factor))) = 255;
      }
      //cv::namedWindow("Output", cv::WINDOW_AUTOSIZE ); cv::imshow("Markers", marker_image);
      cv::namedWindow("CPU Diff", cv::WINDOW_AUTOSIZE ); cv::imshow("CPU Diff", h_diff_image);
      cv::namedWindow("CPU Diff Gray", cv::WINDOW_AUTOSIZE ); cv::imshow("CPU Diff Gray", h_diff_image_gray);
      cv::namedWindow("CPU Corner", cv::WINDOW_AUTOSIZE ); cv::imshow("CPU Corner", corner_image);

      watershed(h_diff_image, marker_image);
      cv::waitKey();

      return 0;
}
