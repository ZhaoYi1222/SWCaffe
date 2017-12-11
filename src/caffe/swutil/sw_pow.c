/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * powt functions: (in SPEs)
 * 1. double: sw_pow_d(double* src, double* dst, int count)
 * 2. float : sw_pow_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_pow_d)();
extern SLAVE_FUN(sw_slave_pow_f)();
typedef union TypeVal_{
  double d;
  float  f;
  int    i;
}TypeVal;
typedef struct powTransPara_st {
  void *src;
  void *dst;
  TypeVal  alpha;
  int count;
}powPara;
// Precondition: already athread_init()
void sw_pow_d(const double* src,const double val, double* dst,const int count) {
  powPara *para = (powPara*)malloc(sizeof(powPara));
  para->src = src;
  para->dst = dst;
  para->alpha.d = val;
  para->count = count;
  athread_spawn(sw_slave_pow_d,para);
  athread_join();
  free(para);
}
void sw_pow_f(const float* src,const float val,float* dst,const int count) {
  powPara *para = (powPara*)malloc(sizeof(powPara));
  para->src = src;
  para->dst = dst;
  para->alpha.f = val;
  para->count = count;
  athread_spawn(sw_slave_pow_f,para);
  athread_join();
  free(para);
}
