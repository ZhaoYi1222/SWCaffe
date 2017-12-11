/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * scalar  functions: (in SPEs)
 * 1. double: sw_scalar_d(double* src,double *scalar, double* dst, int outer_dim,int inner_dim,int scalar_dim)
 * 2. float : sw_scalar_f(float * src,float *scalar, float * dst, int outer_dim,int inner_dim,int scalar_dim)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_add_scalar_d)();
extern SLAVE_FUN(sw_slave_add_scalar_f)();
typedef union TypeVal_{
  double d;
  float  f;
  int    i;
}TypeVal;
typedef struct scalarTransPara_st {
  void *src;
  TypeVal alpha;
  int count;
}scalarPara;
// Precondition: already athread_init()
 void sw_add_scalar_d(double alpha,double *src,int count) {
  scalarPara *para = (scalarPara*)malloc(sizeof(scalarPara));
  para->src = src;
  para->alpha.d = alpha;
  para->count = count;
  athread_spawn(sw_slave_add_scalar_d,para);
  athread_join();
  free(para);
}
void sw_add_scalar_f(float alpha,float *src,int count) {
  scalarPara *para = (scalarPara*)malloc(sizeof(scalarPara));
  para->src = src;
  para->alpha.f = alpha;
  para->count = count;
  athread_spawn(sw_slave_add_scalar_f,para);
  athread_join();
  free(para);
}
