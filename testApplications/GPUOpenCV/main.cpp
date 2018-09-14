#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>


#include "water_color.h"
#include "timer.h"
//#include "SerialRotoscope.h"

using namespace cv::cuda;

int main()
{
      int maxCorners 		= 1000;
      double qualityLevel 	= 0.000001;
      double minDistance 	= 1;
      int blockSize 		= 3;
      bool useHarrisDetector 	= true;
      double k 		      = 0.04;
      //
      // if (cv::cuda::getCudaEnabledDeviceCount() != 0)
      // {
      //       cv::cuda::setDevice(0);
      //       cv::cuda::DeviceInfo dev;
      //       std::cout << "CUDA device : " << dev.name() << std::endl;
      //       std::cout << "Major ver   : "<< dev.majorVersion() << std::endl;
      //       std::cout << "Minor ver   : "<< dev.minorVersion() << std::endl;
      // }

      try
      {
            double t0, t1, t2, t3, t4, t5;

            cv::Size img_sz(2560, 1440);
            cv::Mat          h_image, h_image_back, h_diff_image, h_diff_image_gray, h_out;
            cv::Mat h_corners;

            int factor = 2;
            //cv::cuda::setDevice(0);


            std::vector<cv::Point2f>	corners;

            h_image      = cv::imread("img.png",cv::IMREAD_COLOR);
            h_image_back = cv::imread("back.png",cv::IMREAD_COLOR);

            t2 = getProcessTime();

            cv::absdiff(h_image, h_image_back, h_diff_image);
            cv::cvtColor(h_diff_image, h_diff_image_gray, cv::COLOR_BGR2GRAY);

            // - DownSampling
            if (factor >=2)
            {
                  cv::pyrDown(h_diff_image_gray, h_diff_image_gray);
                  for(int i = 2; i < factor; i = i*2)
                  {
                        cv::pyrDown(h_diff_image_gray, h_diff_image_gray);
                  }
            }

            goodFeaturesToTrack(h_diff_image_gray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

            cv::Mat marker_image = cv::Mat::zeros(h_image.size(), CV_32SC1);
            cv::Mat corner_image = cv::Mat::zeros(h_image.size(), CV_8UC1);

            for (int i = 0; i < corners.size(); i++)
            {
                  marker_image.at<int>(cv::Point(int(corners[i].x * factor), int(corners[i].y * factor))) = i+1;
                  corner_image.at<uchar>(cv::Point(int(corners[i].x * factor), int(corners[i].y * factor))) = 255;
            }
            cv::namedWindow("Output", cv::WINDOW_AUTOSIZE ); cv::imshow("Markers", marker_image);
            cv::namedWindow("Output", cv::WINDOW_AUTOSIZE ); cv::imshow("Markers", corner_image);
            watershed(h_diff_image, marker_image);

            // -- End of CPU computation --
            //
            cv::cuda::GpuMat d_image            (img_sz, CV_8UC3);
            cv::cuda::GpuMat d_diff_image       (img_sz, CV_8UC1);
            cv::cuda::GpuMat d_marker_image     (img_sz, CV_8UC1);
            cv::cuda::GpuMat d_colors           (4, maxCorners, CV_32SC1);
            cv::cuda::GpuMat d_out              (img_sz, CV_8UC3);

            d_image.upload(h_image);
            d_diff_image.upload(h_diff_image);
            d_marker_image.upload(marker_image);

            post_water_seg(d_diff_image, d_marker_image);

            color_palette_image(d_image, d_marker_image, d_colors, d_out);
            d_out.download(h_out);

            t3 = getProcessTime();
            //cv::namedWindow("Markers", cv::WINDOW_AUTOSIZE ); cv::imshow("Markers", marker_image);


            t0 = getProcessTime();
            /*
            cv::cuda::GpuMat d_image            (img_sz, CV_8UC3);
            cv::cuda::GpuMat d_image_back       (img_sz, CV_8UC3);
            cv::cuda::GpuMat d_diff_image_gray  (img_sz, CV_8UC1);
            cv::cuda::GpuMat d_diff_image       (img_sz, CV_8UC1);
            cv::cuda::GpuMat d_marker_image     (img_sz, CV_8UC1);
            cv::cuda::GpuMat d_corners          (1, maxCorners, CV_32FC2);
            cv::cuda::GpuMat d_colors           (4, maxCorners, CV_32SC1);
            cv::cuda::GpuMat d_out              (img_sz, CV_8UC3);

            d_image.upload(h_image);
            d_image_back.upload(h_image_back);

            cv::cuda::absdiff(d_image, d_image_back, d_diff_image);
            cv::cuda::cvtColor(d_diff_image, d_diff_image_gray, cv::COLOR_BGR2GRAY);

            // - DownSampling
            if (factor >=2)
            {
                  cv::cuda::pyrDown(d_diff_image_gray, d_diff_image_gray);
                  for(int i = 2; i < factor; i = i*2)
                  {
                        cv::cuda::pyrDown(d_diff_image_gray, d_diff_image_gray);
                  }
            }

            cv::Ptr<cv::cuda::CornersDetector> cd = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k);

            cd->detect(d_diff_image_gray, d_corners);

            // -- Migrate Corner positions back to CPU --

            d_corners.download(h_corners);

            cv::Mat marker_image = cv::Mat::zeros(h_image.size(), CV_32SC1);
            //cv::Mat corner_image = cv::Mat::zeros(h_image.size(), CV_8UC1);
            for (int i = 0; i < h_corners.cols - 1; i++)
            {
                  //corner_image.at<uchar>(  int(h_corners.at<cv::Vec2f>(i)[1]*2), int(h_corners.at<cv::Vec2f>(i)[0]*2)  ) = 255;
                  marker_image.at<int>(  int(h_corners.at<cv::Vec2f>(i)[1]*2), int(h_corners.at<cv::Vec2f>(i)[0]*2)  ) = i + 1;
            }

            //cv::Mat h_diff_image;
            d_diff_image.download(h_diff_image);
            watershed(h_diff_image, marker_image);

            d_marker_image.upload(marker_image);

            // -- End of CPU computation --

            post_water_seg(d_diff_image, d_marker_image);

            color_palette_image(d_image, d_marker_image, d_colors, d_out);

            //cv::Mat h_out;
            d_out.download(h_out);

            */
            t1 = getProcessTime();

            std::cout << "GPU Frame time : " << t1 - t0 << " seconds" << std::endl;
            std::cout << "CPU Frame time : " << t3 - t2 << " seconds" << std::endl;


            //std::cout << "Current Section time : " << t3 - t2 << " seconds" << std::endl;

            //cv::namedWindow("Markers", cv::WINDOW_AUTOSIZE ); cv::imshow("Markers", corner_image);
            cv::namedWindow("Output", cv::WINDOW_AUTOSIZE ); cv::imshow("Output", h_out);
            cv::waitKey();

      }
      catch(const cv::Exception& ex)
      {
            std::cout << "Error: " << ex.what() << std::endl;
      }

      return 0;
}
