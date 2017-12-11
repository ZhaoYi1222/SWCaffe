#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer
#define BUFFSIZE 2*1024
#define SIMDSIZE 4
#define SIMDTYPED doublev4
#define SIMDTYPEF floatv4
#define SPNUM 64

__thread_local dma_desc dma_get_src, dma_put_dst;

typedef struct _tagWeightsAdd
{
  void * src,*dst;
  int num,count;
}WeightsAdd;
void sw_slave_weights_add_f(WeightsAdd * para){
  SIMDTYPEF vsrc1,vsrc2;
  int id = athread_get_id(-1);
  int num = para->num;
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;
  float* local_src1 = (float*)ldm_malloc(BUFFSIZE*sizeof(float));
  float* local_src2 = (float*)ldm_malloc(BUFFSIZE*sizeof(float));
  float* src_ptr = (float*)para->src;
  float* dst_ptr = (float*)para->dst;
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,BUFFSIZE*sizeof(float));
  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(float));
  
  int n = 0;

  for(off = start; off+BUFFSIZE-1<end; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src1));
    dma_wait(&replyget, 1); replyget = 0;
    
    for(n= 1;n < num;n++){
      dma(dma_get_src, (long)(src_ptr+n*count + off), (long)(local_src2));
      dma_wait(&replyget, 1); replyget = 0;
      for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
         simd_load(vsrc1,&local_src1[i]);
         simd_load(vsrc2,&local_src2[i]);
         vsrc1 = vsrc1 + vsrc2;
         simd_store(vsrc1,&local_src1[i]);
       }
    }
    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_src1));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_src,(local_count-off)*sizeof(float));
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src1));
    dma_wait(&replyget, 1); replyget = 0;

    for(n= 1;n < num;n++){
      dma(dma_get_src, (long)(src_ptr+n*count + off), (long)(local_src2));
      dma_wait(&replyget, 1); replyget = 0;
      for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
         simd_load(vsrc1,&local_src1[i]);
         simd_load(vsrc2,&local_src2[i]);
         vsrc1 = vsrc1 + vsrc2;
         simd_store(vsrc1,&local_src1[i]);
      }
      for(;i<local_count-off;i++) {
         local_src1[i]=local_src1[i] + local_src2[i];
      }
    }
    dma_set_size(&dma_put_dst,(local_count-off)*sizeof(float));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_src1));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, BUFFSIZE*sizeof(float));
  ldm_free(local_src2, BUFFSIZE*sizeof(float));
}
/*
void sw_slave_weights_add_d(WeightsAdd * para){
  SIMDTYPED vsrc1,vsrc2;
  int id = athread_get_id(-1);
  int num = para->num;
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;
  double* local_src1 = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  double* local_src2 = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  double* src_ptr = (double*)para->src;
  double* dst_ptr = (double*)para->dst;
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,BUFFSIZE*sizeof(double));
  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(double));
  
  int n = 0;
  for(off = start; off+BUFFSIZE-1<end; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src1));
    dma_wait(&replyget, 1); replyget = 0;
    
    for(n= 1;n < num;n++){
      dma(dma_get_src, (long)(src_ptr+n*count + off), (long)(local_src2));
      dma_wait(&replyget, 1); replyget = 0;
      for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
         simd_load(vsrc1,&local_src1[i]);
         simd_load(vsrc2,&local_src2[i]);
         vsrc1 = vsrc1 + vsrc2;
         simd_store(vsrc1,&local_src1[i]);
       }
    }
    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_src1));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_src,(local_count-off)*sizeof(double));
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src1));
    dma_wait(&replyget, 1); replyget = 0;

    for(n= 1;n < num;n++){
      dma(dma_get_src, (long)(src_ptr+n*count + off), (long)(local_src2));
      dma_wait(&replyget, 1); replyget = 0;
      for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
         simd_load(vsrc1,&local_src1[i]);
         simd_load(vsrc2,&local_src2[i]);
         vsrc1 = vsrc1 + vsrc2;
         simd_store(vsrc1,&local_src1[i]);
      }
      for(;i<local_count-off;i++) {
         local_src1[i]=local_src1[i] + local_src2[i];
      }
    }
    dma_set_size(&dma_put_dst,(local_count-off)*sizeof(double));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_src1));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src1, BUFFSIZE*sizeof(double));
  ldm_free(local_src2, BUFFSIZE*sizeof(double));
}
*/
