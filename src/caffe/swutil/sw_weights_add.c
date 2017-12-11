#include "caffe/util/sw_dnn.h"
#include "athread.h"

typedef struct _tagWeightsAdd
{
  void * src,*dst;
  int num,count;
}WeightsAdd;

extern SLAVE_FUN(sw_slave_weights_add_f)();
//extern SLAVE_FUN(sw_slave_weights_add_d)();

void sw_weights_add_f(float *dst,const float *src,int num,int count){
   
  WeightsAdd *para = (WeightsAdd*)malloc(sizeof(WeightsAdd));
  para->src = src;
  para->dst = dst;
  para->num = num;
  para->count = count;
  athread_spawn(sw_slave_weights_add_f,para);
  athread_join();
  free(para);
}
void sw_weights_add_d(double *dst,const double *src,int num,int count)
{
  WeightsAdd *para = (WeightsAdd*)malloc(sizeof(WeightsAdd));
  para->src = src;
  para->dst = dst;
  para->num = num;
  para->count = count;
  //athread_spawn(sw_slave_weights_add_d,para);
  //athread_join();
  free(para);
}
