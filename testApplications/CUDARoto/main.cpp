#include <opencv2/opencv.hpp>

#include "Roto.h"

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
      unsigned char *diff, *diff_gray;
      size_t _channel_size = sizeof(unsigned char) * h_image.cols * h_image.rows;
      diff = (unsigned char *)malloc(_channel_size * 3);
      diff_gray = (unsigned char *)malloc(_channel_size);

      // First CUDA Kernel call function performs both absdiff and cvtColor in one.
      diff_and_convert(v_image.data(),
                       v_back.data(),
                       diff,
                       diff_gray,
                       h_image.cols,
                       h_image.rows,
                       _channel_size);

      // -----------------------------------------------------------------------
      //
      // Purely to be used to check the output of the data from the GPU kernels
      //
      // -----------------------------------------------------------------------

      cv::Mat out = cv::Mat(h_image.rows, h_image.cols, CV_8UC3);
      cv::Mat o_diff = cv::Mat(h_image.rows, h_image.cols, CV_8UC3);
      cv::Mat o_diff_gray = cv::Mat(h_image.rows, h_image.cols, CV_8UC1);

      int indx = 0;
      for (int i = 0; i < h_image.rows; i++)
      {
            for (int j = 0; j < h_image.cols; j++)
            {
                  out.at<cv::Vec3b>(i,j) = cv::Vec3b(v_image[indx],
                                                     v_image[indx+1],
                                                     v_image[indx+2]);
                  indx += 3;
            }
      }

      indx = 0;
      for (int i = 0; i < h_image.rows; i++)
      {
            for (int j = 0; j < h_image.cols; j++)
            {
                  o_diff.at<cv::Vec3b>(i,j) = cv::Vec3b(diff[indx],
                                                     diff[indx+1],
                                                     diff[indx+2]);
                  indx += 3;
            }
      }

      indx = 0;
      for (int i = 0; i < h_image.rows; i++)
      {
            for (int j = 0; j < h_image.cols; j++)
            {
                  o_diff_gray.at<uchar>(i,j) = diff_gray[indx];
                  indx++;
            }
      }
      cv::namedWindow("Image", cv::WINDOW_AUTOSIZE ); cv::imshow("Image", out);
      cv::namedWindow("Difference", cv::WINDOW_AUTOSIZE ); cv::imshow("Difference", o_diff);
      cv::namedWindow("Diff Gray", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Gray", o_diff_gray);

      cv::waitKey();

      return 0;
}
