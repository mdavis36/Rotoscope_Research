
#include "SerialRotoscope.h"

int main(int argc, char *argv[])
{
      SerialRotoscope serial_r;
      //OMPRotoscope omp_r;

      serial_r.runRotoscope(argc, argv);
      return 0;


}
