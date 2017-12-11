/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * expt functions: (in SPEs)
 * 1. double: sw_exp_d(double* src, double* dst, int count)
 * 2. float : sw_exp_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_exp_d)();
extern SLAVE_FUN(sw_slave_exp_f)();
typedef struct expTransPara_st {
  void *src;
  void *dst;
  int count;
}expPara;
// Precondition: already athread_init()
void sw_exp_d(const double* src, double* dst,const int count) {
  expPara *para = (expPara*)malloc(sizeof(expPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_exp_d,para);
  athread_join();
  free(para);
}
void sw_exp_f(const float* src, float* dst,const int count) {
  expPara *para = (expPara*)malloc(sizeof(expPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_exp_f,para);
  athread_join();
  free(para);
}
