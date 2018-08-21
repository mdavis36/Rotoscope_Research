#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

using namespace cv::cuda;

int main()
{
      std::cout << "Hello World\n";
      if (cv::cuda::getCudaEnabledDeviceCount() != 0)
      {
            cv::cuda::setDevice(0);
            cv::cuda::DeviceInfo dev;
            std::cout << "CUDA device : " << dev.name() << std::endl;
            std::cout << "Major ver   : "<< dev.majorVersion() << std::endl;
            std::cout << "Minor ver   : "<< dev.minorVersion() << std::endl;
      }

      try
      {
            int factor = 2;

            cv::Mat          h_image, h_image_back;
            cv::cuda::GpuMat d_image, d_image_back, d_diff_image, d_diff_image_gray, d_diff_image_gray_ds;

            h_image      = cv::imread("img.png",cv::IMREAD_COLOR);
            h_image_back = cv::imread("back.png",cv::IMREAD_COLOR);

            d_image.upload(h_image);
            d_image_back.upload(h_image_back);

            cv::cuda::absdiff(d_image, d_image_back, d_diff_image);

            cv::cuda::cvtColor(d_diff_image, d_diff_image_gray, cv::COLOR_BGR2GRAY);

            // - DownSampling
            if (factor >=2)
            {
                  cv::cuda::pyrDown(d_diff_image_gray, d_diff_image_gray_ds);
                  for(int i = 2; i < factor; i = i*2)
                  {
                        cv::cuda::pyrDown(d_diff_image_gray_ds, d_diff_image_gray_ds);
                  }
            }
            else
            {
                  d_diff_image_gray_ds = cv::cuda::GpuMat(d_diff_image_gray);
            }


            cv::Mat h_diff_image, h_diff_image_gray, h_diff_image_gray_ds;
            d_diff_image.download(h_diff_image);
            // d_diff_image_gray.download(h_diff_image_gray);
            // d_diff_image_gray_ds.download(h_diff_image_gray_ds);
            // cv::namedWindow("Diff Image", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Image", h_diff_image);
            // cv::namedWindow("Diff Gray Image", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Gray Image", h_diff_image_gray);
            // cv::namedWindow("Diff Gray Image DS", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Gray Image DS", h_diff_image_gray_ds);

            int maxCorners 		= 1000;
            double qualityLevel 	= 0.000001;
            double minDistance 	= 1;
            int blockSize 		= 3;
            bool useHarrisDetector 	= false;
            double k 		      = 0.04;

            cv::cuda::GpuMat d_corners(1,maxCorners,CV_32FC2);
            cv::Ptr<cv::cuda::CornersDetector> cd = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, k);
            cd->detect(d_diff_image_gray_ds, d_corners);



            // -- Migrate Corner positions back to CPU --

            cv::Mat h_corners;
            d_corners.download(h_corners);

            cv::Mat marker_image = cv::Mat::zeros(h_image.size(), CV_32SC1);
            cv::Mat corner_image = cv::Mat::zeros(h_image.size(), CV_8UC1);
            for (int i = 0; i < h_corners.cols - 1; i++)
            {
                  corner_image.at<uchar>(  int(h_corners.at<cv::Vec2f>(i)[1]*2), int(h_corners.at<cv::Vec2f>(i)[0]*2)  ) = 255;
                  marker_image.at<int>(  int(h_corners.at<cv::Vec2f>(i)[1]*2), int(h_corners.at<cv::Vec2f>(i)[0]*2)  ) = i + 1;
            }

            cv::cuda::GpuMat d_markers;
            d_markers.upload(marker_image);

            watershed(h_diff_image, marker_image);





            cv::namedWindow("Markers", cv::WINDOW_AUTOSIZE ); cv::imshow("Markers", marker_image);
            cv::waitKey();

      }
      catch(const cv::Exception& ex)
      {
            std::cout << "Error: " << ex.what() << std::endl;
      }

      return 0;
}
