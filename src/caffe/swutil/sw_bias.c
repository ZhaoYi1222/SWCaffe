#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_bias_f)();

typedef struct biasPara_ {
  void *A,*B,*C;
  int num,M,N;
}biasPara;

void sw_bias_f(const int num,const int M,const int N,float *A,float*B,float*C)
{
   biasPara *para = (biasPara*)malloc(sizeof(biasPara));
   para->num = num;
   para->M = M;
   para->N = N;
   para->A = A;
   para->B = B;
   para->C = C;
   athread_spawn(sw_slave_bias_f,para);
   athread_join();
   free(para);
}
void sw_bias_d(const int num,const int M,const int N,double *A,double*B,double*C)
{
   biasPara *para = (biasPara*)malloc(sizeof(biasPara));
   para->num = num;
   para->M = M;
   para->N = N;
   para->A = A;
   para->B = B;
   para->C = C;
   //athread_spawn(sw_slave_bias_f,para);
   //athread_join();
   free(para);
}
