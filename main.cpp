
#include "SerialRotoscope.h"
#include "OMPRotoscope.h"

int main(int argc, char *argv[])
{
      SerialRotoscope serial_r;
      OMPRotoscope omp_r;

      serial_r.runRotoscope(argc, argv);
      //omp_r.runRotoscope(argc, argv);
      return 0;


}
