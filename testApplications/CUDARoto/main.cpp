#include <opencv2/opencv.hpp>

#include "Roto.h"


int main()
{
      int num_rows, num_cols;

      cv::Mat h_image = cv::imread("img.png",cv::IMREAD_COLOR);
      cv::Mat h_back = cv::imread("back.png",cv::IMREAD_COLOR);

      std::vector<unsigned char> v_image, v_back, diff, diff_gray;

      if (h_image.isContinuous())
            v_image.assign(h_image.datastart, h_image.dataend);

      if (h_back.isContinuous())
            v_back.assign(h_image.datastart, h_image.dataend);

      //unsigned char *diff, *diff_gray;

      diff_and_convert(v_image.data(),
                       v_back.data(),
                       diff.data(),
                       diff_gray.data(),d
                       h_image.cols,
                       h_image.rows);



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
      cv::namedWindow("Image", cv::WINDOW_AUTOSIZE ); cv::imshow("Image", out);
      cv::namedWindow("Difference", cv::WINDOW_AUTOSIZE ); cv::imshow("Difference", o_diff);
      cv::namedWindow("Diff Gray", cv::WINDOW_AUTOSIZE ); cv::imshow("Diff Gray", o_diff_gray);

      cv::waitKey();

      return 0;
}
