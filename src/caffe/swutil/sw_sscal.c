/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * sscal  functions: (in SPEs)
 * 1. double: sw_sscal_d(double* src,double *sscal, double* dst, int outer_dim,int inner_dim,int sscal_dim)
 * 2. float : sw_sscal_f(float * src,float *sscal, float * dst, int outer_dim,int inner_dim,int sscal_dim)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_sscal_d)();
extern SLAVE_FUN(sw_slave_sscal_f)();
typedef union TypeVal_{
  double d;
  float  f;
  int    i;
}TypeVal;
typedef struct sscalTransPara_st {
  void *src;
  void *dst;
  TypeVal  alpha;
  int count;
}sscalPara;
// Precondition: already athread_init()
void sw_sscal_d(const double *src,const double alpha,double *dst,const int count) {
  sscalPara *para = (sscalPara*)malloc(sizeof(sscalPara));
  para->src = src;
  para->dst = dst;
  para->alpha.d = alpha;
  para->count = count;
  athread_spawn(sw_slave_sscal_d,para);
  athread_join();
  free(para);
}
void sw_sscal_f(const float *src,const float alpha,float *dst,const int count) {
  sscalPara *para = (sscalPara*)malloc(sizeof(sscalPara));
  para->src = src;
  para->dst = dst;
  para->alpha.f = alpha;
  para->count = count;
  athread_spawn(sw_slave_sscal_f,para);
  athread_join();
  free(para);
}
