#pragma once
#ifndef _TIMES_H
#define _TIMES_H
 
#ifdef _WIN32
#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>
 
int gettimeofday(struct timeval* t,void* timezone);
 
// from linux's sys/times.h
 
//#include <features.h>
 
#define __need_clock_t
#include <time.h>
 
 
/* Structure describing CPU time used by a process and its children.  */
struct tms
  {
    clock_t tms_utime;          /* User CPU time.  */
    clock_t tms_stime;          /* System CPU time.  */
 
    clock_t tms_cutime;         /* User CPU time of dead children.  */
    clock_t tms_cstime;         /* System CPU time of dead children.  */
  };
 
/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t times (struct tms *__buffer);
 
typedef long long suseconds_t ;

#define CLOCK_REALTIME 0
#define BILLION                             (1E9)



static BOOL g_first_time = 1;
static LARGE_INTEGER g_counts_per_sec;
inline int clock_gettime(int dummy, struct timespec *ct)
{
    LARGE_INTEGER count;

    if (g_first_time)
    {
        g_first_time = 0;

        if (0 == QueryPerformanceFrequency(&g_counts_per_sec))
        {
            g_counts_per_sec.QuadPart = 0;
        }
    }

    if ((NULL == ct) || (g_counts_per_sec.QuadPart <= 0) ||
            (0 == QueryPerformanceCounter(&count)))
    {
        return -1;
    }

    ct->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
	ct->tv_nsec = (long)(((count.QuadPart % g_counts_per_sec.QuadPart) * BILLION) / g_counts_per_sec.QuadPart);

    return 0;
}

#endif
#endif