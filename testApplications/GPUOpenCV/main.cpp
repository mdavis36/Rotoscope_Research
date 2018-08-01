#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>

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

      return 0;
}
