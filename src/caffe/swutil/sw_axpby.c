/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * axpbytify  functions: (in SPEs)
 * 1. double: sw_axpby_d(double* src1,double *src2, double* dst, int count)
 * 2. float : sw_axpby_f(float * src1,float *src2, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

//extern SLAVE_FUN(sw_slave_axpby_d)();
extern SLAVE_FUN(sw_slave_axpby_f)();
typedef union TypeVal_{
  double d;
  float  f;
  int    i;
}TypeVal;
typedef struct axpbyTransPara_st {
  void *src;
  void *dst;
  TypeVal alpha;
  TypeVal beta;
  int count;
}axpbyPara;
// Precondition: already athread_init()
void sw_axpby_d(const double alpha,const double *src,const double beta, double* dst,const int count) {
  axpbyPara *para = (axpbyPara*)malloc(sizeof(axpbyPara));
  para->alpha.d = alpha;
  para->beta.d = beta;
  para->src = src;
  para->dst = dst;
  para->count = count;
  //athread_spawn(sw_slave_axpby_d,para);
  //athread_join();
  free(para);
}
void sw_axpby_f(const float alpha,const float *src,const float beta,float* dst,const int count) {
  axpbyPara *para = (axpbyPara*)malloc(sizeof(axpbyPara));
  para->alpha.f = alpha;
  para->beta.f = beta;
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_axpby_f,para);
  athread_join();
  free(para);
}
