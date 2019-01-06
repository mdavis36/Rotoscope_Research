#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include "Roto.cuh"

int timeval_subtract (timeval *result, timeval *x, timeval *y)
{
      /* Perform the carry for the later subtraction by updating y. */
      if (x->tv_usec < y->tv_usec) {
            int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
            y->tv_usec -= 1000000 * nsec;
            y->tv_sec += nsec;
      }
      if (x->tv_usec - y->tv_usec > 1000000) {
            int nsec = (y->tv_usec - x->tv_usec) / 1000000;
            y->tv_usec += 1000000 * nsec;
            y->tv_sec -= nsec;
      }
      /* Compute the time remaining to wait.
         tv_usec is certainly positive. */
      result->tv_sec = x->tv_sec - y->tv_sec;
      result->tv_usec = x->tv_usec - y->tv_usec;

      /* Return 1 if result is negative. */
      return x->tv_sec < y->tv_sec;
}

int main(int argc, char *argv[])
{
      if (argc < 2)
      {
            std::cout << "Please provide .mp4 file." << std::endl;
            return 0;
      }

      CUDA_Rotoscope cr(argv[1]);

      timeval tv, tv1, tv0;
      gettimeofday(&tv, NULL);
      gettimeofday(&tv0, NULL);

      cr.processVideo();

      gettimeofday(&tv1, NULL);
      gettimeofday(&tv, NULL);

      timeval_subtract(&tv, &tv1, &tv0);
      printf("Compute Time    : %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
      return 0;
}
