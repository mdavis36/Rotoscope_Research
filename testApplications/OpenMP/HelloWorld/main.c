#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 16

static long num_steps = 100000000;
double step;


int main()
{
      #pragma omp parallel num_threads(4)
      {
            int ID = omp_get_thread_num();
            printf("hello (%d)", ID);
            printf("world (%d)\n", ID);
      }

      int i;
      double x, pi_s, pi_p, sum = 0.0;
      double t1, t2, t3, t4;



      // Serial pi
      t1 = omp_get_wtime();

      omp_set_num_threads(NUM_THREADS);
      omp_set_schedule(omp_sched_static, num_steps / omp_get_max_threads());
      step = 1.0/(double)num_steps;

      #pragma omp parallel for reduction (+:sum) schedule(runtime)
      for (i = 0; i < num_steps; i++)
      {
            double x = (i + 0.5) * step;
            sum = sum + 4.0 / (1.0+x*x);
      }
      pi_s = step * sum;

      t2 = omp_get_wtime();



      // Parallel pi
      t3 = omp_get_wtime();

      omp_set_num_threads(NUM_THREADS);
      sum = 0.0;
      int delta = num_steps / NUM_THREADS;

      #pragma omp parallel
      {
            double x, temp_sum = 0.0;
            int i;
            int t_id = omp_get_thread_num();
            int t_start = t_id * delta;
            int t_end = (t_id + 1) * delta;
            if (t_end > num_steps) t_end = num_steps;

            for (i = t_start; i < t_end; i++)
            {
                  x = (i + 0.5) * step;
                  temp_sum = temp_sum + 4.0 / (1.0+x*x);
            }
      #pragma omp atomic
            pi_p += temp_sum * step;
      }

      t4 = omp_get_wtime();




      printf("pi   (omp for) : %lf\n", pi_s);
      printf("pi   (SPMD)    : %lf\n", pi_p);
      printf("time (omp for) : %lf\n", t2 - t1);
      printf("time (SPMD)    : %lf\n", t4 - t3);

}
