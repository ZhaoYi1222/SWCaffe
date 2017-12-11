#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_trans_f)();
//extern SLAVE_FUN(sw_slave_trans_d)();

typedef struct transPara_st {
  void *src;
  void *dst;
  int num,K,N;
}transPara;

void sw_trans_f(const int num,const int row,const int col,const float * src,float * dst)
{
   int SIMDSIZE = 4;
   int i=0,j=0,n=0,k=0;
   int M = num * row;
   int N = col - (col%SIMDSIZE);
   int size = row * col;
   if(M<4)
   {
     for(i = 0;i < M;i++)
     {
       n = i / row;
       k = i % row;
       for(j = 0;j < col;j++)
       {
         dst[n*size + j*row + k] = src[n*size + k*col +j];
       }
    }
   }
   transPara *para = (transPara*)malloc(sizeof(transPara));
   para->num = num;
   para->K = row;
   para->N = col;
   para->src = src;
   para->dst = dst;
   athread_spawn(sw_slave_trans_f,para);
   for(i = M - (M%SIMDSIZE);i < M;i++)
   {
      n = i / row;
      k = i % row;
      for(j = 0;j < col;j++)
      {
        dst[n*size + j*row + k] = src[n*size + k*col +j];
      }
   }
   for(i = 0;i < M;i++)
   {
      n = i / row;
      k = i % row;
      for(j = N;j < col;j++)
      {
        dst[n*size + j*row + k] = src[n*size + k*col +j];
      }
   }
   athread_join();
   free(para);
}
void sw_trans_d(const int num,const int row,const int col,const double * src,double * dst)
{
   int SIMDSIZE = 4;
   int i=0,j=0,n=0,k=0;
   int M = num * row;
   int N = col - (col%SIMDSIZE);
   int size = row * col;
   if(M<4)
   {
     for(i = 0;i < M;i++)
     {
       n = i / row;
       k = i % row;
       for(j = 0;j < col;j++)
       {
         dst[n*size + j*row + k] = src[n*size + k*col +j];
       }
    }
   }
   transPara *para = (transPara*)malloc(sizeof(transPara));
   para->num = num;
   para->K = row;
   para->N = col;
   para->src = src;
   para->dst = dst;
   //athread_spawn(sw_slave_trans_d,para);
   for(i = M - (M%SIMDSIZE);i < M;i++)
   {
      n = i / row;
      k = i % row;
      for(j = 0;j < col;j++)
      {
        dst[n*size + j*row + k] = src[n*size + k*col +j];
      }
   }
   for(i = 0;i < M;i++)
   {
      n = i / row;
      k = i % row;
      for(j = N;j < col;j++)
      {
        dst[n*size + j*row + k] = src[n*size + k*col +j];
      }
   }
   //athread_join();
   free(para);
}

