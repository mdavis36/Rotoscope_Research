#include "timer.h"

funcTimer::funcTimer(){
      last = {0,0};
      acc_time = {0,0};
}


Timer::Timer(string n)
{
      name  = n;

}

Timer::~Timer()
{

}

void Timer::startTimer(string name)
{
      struct timespec temp;
      clock_gettime(CLOCK_REALTIME, &temp);

      if (times_map.count(name) <= 0) // not yet created
      {
            funcTimer ft;
            ft.last = temp;
            times_map.insert(pair<string,funcTimer>(name, ft));
      }
      else
      {
            auto it = times_map.find(name);
            if (it != times_map.end())
                  it->second.last = temp;
      }

}

void Timer::endTimer(string name)
{

      if (times_map.count(name) <= 0) // not yet created
      {
            cout << "ERROR : " << name << " timer has not yet been initialized."<<endl;
            exit(-1);
      }
      else
      {
            struct timespec temp;
            clock_gettime(CLOCK_REALTIME, &temp);

            auto it = times_map.find(name);
            if (it != times_map.end())
                  temp = diff_time(it->second.last, temp);
                  it->second.acc_time = add_time(it->second.acc_time, temp);
                  it->second.count++;
      }
}

void Timer::calcFinalTime()
{


}

void Timer::printFinalTimeData()
{

      cout << "- " << name <<endl;
      for(auto const & x : times_map)
      {
            cout << "\t-------------------------" << endl;
            cout << "\tFunc Name  : " << x.first << endl;
            cout << "\tAccum Time : "<< x.second.acc_time.tv_sec << "." << x.second.acc_time.tv_nsec << endl;
            cout << "\tAVG R-Time : ";
            struct timespec avg = divByInt(x.second.acc_time , x.second.count);
            printf("%lld.%.9ld\n", (long long)avg.tv_sec, avg.tv_nsec);
            cout << "\tCall Count : " << x.second.count << endl << endl;
      }
}


double Timer::timespecToDouble(struct timespec t)
{
      return (double)t.tv_sec + ((double)t.tv_nsec / (double)BILLION);
}
