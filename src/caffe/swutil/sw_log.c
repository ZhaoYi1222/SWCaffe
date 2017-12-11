/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * logt functions: (in SPEs)
 * 1. double: sw_log_d(double* src, double* dst, int count)
 * 2. float : sw_log_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_log_d)();
extern SLAVE_FUN(sw_slave_log_f)();
typedef struct logTransPara_st {
  void *src;
  void *dst;
  int count;
}logPara;
// Precondition: already athread_init()
void sw_log_d(const double* src, double* dst,const int count) {
  logPara *para = (logPara*)malloc(sizeof(logPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_log_d,para);
  athread_join();
  free(para);
}
void sw_log_f(const float* src, float* dst,const int count) {
  logPara *para = (logPara*)malloc(sizeof(logPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_log_f,para);
  athread_join();
  free(para);
}
