#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_gemm_im2col_f)();
extern SLAVE_FUN(sw_slave_gemm_col2im_f)();
//extern SLAVE_FUN(sw_slave_gemm_im2col_d)();
//extern SLAVE_FUN(sw_slave_gemm_col2im_d)();

typedef struct gemmTransPara_st {
  void *src;
  void *dst;
  int num,M,N,group;
}gemmTransPara;

void sw_gemm_im2col_f(float *dst,const float *src,int M,int N,int num,int group)
{
   gemmTransPara *para = (gemmTransPara*)malloc(sizeof(gemmTransPara));
   para->num = num;
   para->M = M;
   para->N = N;
   para->group = group;
   para->src = src;
   para->dst = dst;
   athread_spawn(sw_slave_gemm_im2col_f,para);
   athread_join();
   free(para);
}
void sw_gemm_im2col_d(double *dst,const double *src,int M,int N,int num,int group)
{
   gemmTransPara *para = (gemmTransPara*)malloc(sizeof(gemmTransPara));
   para->num = num;
   para->M = M;
   para->N = N;
   para->group = group;
   para->src = src;
   para->dst = dst;
   //athread_spawn(sw_slave_gemm_im2col_d,para);
   //athread_join();
   free(para);
}
void sw_gemm_col2im_f(float *dst,const float *src,int M,int N,int num){
   gemmTransPara *para = (gemmTransPara*)malloc(sizeof(gemmTransPara));
   para->num = num;
   para->M = M;
   para->N = N;
   para->src = src;
   para->dst = dst;
   athread_spawn(sw_slave_gemm_col2im_f,para);
   athread_join();
   free(para);
}
void sw_gemm_col2im_d(double *dst,const double *src,int M,int N,int num){
   gemmTransPara *para = (gemmTransPara*)malloc(sizeof(gemmTransPara));
   para->num = num;
   para->M = M;
   para->N = N;
   para->src = src;
   para->dst = dst;
   //athread_spawn(sw_slave_gemm_col2im_d,para);
   //athread_join();
   free(para);
}

