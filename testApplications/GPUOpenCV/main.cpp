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
            d_diff_image_gray.download(h_diff_image_gray);
            d_diff_image_gray_ds.download(h_diff_image_gray_ds);
            cv::namedWindow("Diff Image", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Image", h_diff_image);
            cv::namedWindow("Diff Gray Image", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Gray Image", h_diff_image_gray);
            cv::namedWindow("Diff Gray Image DS", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Gray Image DS", h_diff_image_gray_ds);
            cv::waitKey();

      }
      catch(const cv::Exception& ex)
      {
            std::cout << "Error: " << ex.what() << std::endl;
      }

      return 0;
}
