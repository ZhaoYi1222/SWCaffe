/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * axpytify  functions: (in SPEs)
 * 1. double: sw_axpy_d(double* src1,double *src2, double* dst, int count)
 * 2. float : sw_axpy_f(float * src1,float *src2, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

//extern SLAVE_FUN(sw_slave_axpy_d)();
extern SLAVE_FUN(sw_slave_axpy_f)();
typedef union TypeVal_{
  double d;
  float  f;
  int    i;
}TypeVal;
typedef struct axpyTransPara_st {
  void *src;
  void *dst;
  TypeVal alpha;
  int count;
}axpyPara;
// Precondition: already athread_init()
void sw_axpy_d(const double alpha,const double *src, double* dst,const int count) {
  axpyPara *para = (axpyPara*)malloc(sizeof(axpyPara));
  para->alpha.d = alpha;
  para->src = src;
  para->dst = dst;
  para->count = count;
  //athread_spawn(sw_slave_axpy_d,para);
  //athread_join();
  free(para);
}
void sw_axpy_f(const float alpha,const float *src, float* dst,const int count) {
  axpyPara *para = (axpyPara*)malloc(sizeof(axpyPara));
  para->alpha.f = alpha;
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_axpy_f,para);
  athread_join();
  free(para);
}
