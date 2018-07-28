#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <string.h>
#include <iostream>
#include <map>

#include "Utils.h"

#define BILLION 1000000000

using namespace std;
using namespace timing;

// struct funcTimer{
//       struct timespec last;
//       struct timespec acc_time;
//       int count = 0;
// } ;

class funcTimer
{
public:
      funcTimer();
      struct timespec last;
      struct timespec acc_time;
      int count = 0;
};

class Timer
{
public:

      string name;
      Timer(string n);
      ~Timer();

      void addNewTimer(string name);

      void startTimer(string name);

      void endTimer(string name);

      void calcFinalTime();

      void printFinalTimeData();
private:
      map<string, funcTimer> times_map;

      double timespecToDouble(struct timespec t);
};



#endif
