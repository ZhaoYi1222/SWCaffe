#include <slave.h>
#include <simd.h>
#include <dma.h>

__thread_local dma_desc dma_gemm_get,dma_gemm_put;

typedef struct _gemmTransPara {
  void *src;
  void *dst;
  int num,M,N,group;
}gemmTransPara;

inline void mb()
{
    asm volatile("":::"memory");
    asm volatile("memb");
}
void sw_slave_gemm_im2col_f(gemmTransPara *para) {
  const int  max_size = 48 * 1024;
  const int  SPNUM = 64; 
  int i,off,n,m;
  int id = athread_get_id(-1);
  int num = para->num;
  int M = para->M;
  int N = para->N;
  int group = para->group;
  int count = M*num;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;
  if(local_count <1) return;
  int col_size = N;
  int max_rows_num = max_size / (col_size * sizeof(float));
  while(max_rows_num < 1)
  {
    col_size = col_size >>1;
    max_rows_num = max_size / (col_size * sizeof(float));
  }
  float * local_src = (float *)ldm_malloc(col_size*sizeof(float ));
  float * src_ptr = (float *)(para->src);
  float * dst_ptr = (float *)(para->dst);
  volatile int replyget=0,replyput=0;
  // DMA settings
	dma_set_op(&dma_gemm_get, DMA_GET);
	dma_set_mode(&dma_gemm_get, PE_MODE);
	dma_set_reply(&dma_gemm_get, &replyget);	
	//dma_set_size(&dma_gemm_get, col_size*sizeof(float));

	dma_set_op(&dma_gemm_put, DMA_PUT);
	dma_set_mode(&dma_gemm_put, PE_MODE);
	dma_set_reply(&dma_gemm_put, &replyput);
	//dma_set_size(&dma_gemm_put, col_size*sizeof(float));
	int left_col_size,left_size,g,offset;
  for(off = start; off < end; off ++)
  {
    n = off / M;
    m = off % M;
    g = m / (M/group);
    m = m % (M/group);
    offset = g*num*M*N/group;
	  dma_set_size(&dma_gemm_get, col_size*sizeof(float));
	  dma_set_size(&dma_gemm_put, col_size*sizeof(float));
		for(i = 0;i+col_size-1 < N;i += col_size)
    {
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + offset +  m*N*num + n*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
    if(i < N){
      left_col_size = N % col_size;
	    dma_set_size(&dma_gemm_get, left_col_size*sizeof(float));
      
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
	    
      dma_set_size(&dma_gemm_put, left_col_size*sizeof(float));

		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + offset +  m*N*num + n*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
  }
  ldm_free(local_src, col_size*sizeof(float));
}
void sw_slave_gemm_col2im_f(gemmTransPara *para) {
  const int  max_size = 48 * 1024;
  const int  SPNUM = 64; 
  int i,off,n,m;
  int id = athread_get_id(-1);
  int num = para->num;
  int M = para->M;
  int N = para->N;
  int count = M * num;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;

  if(local_count <1) return;
  int col_size = N;
  int max_rows_num = max_size / (col_size * sizeof(float));
  while(max_rows_num < 1)
  {
    col_size = col_size >>1;
    max_rows_num = max_size / (col_size * sizeof(float));
  }
  float * local_src = (float *)ldm_malloc(col_size*sizeof(float ));
  float * src_ptr = (float *)(para->src);
  float * dst_ptr = (float *)(para->dst);
  volatile int replyget=0,replyput=0;
  // DMA settings
	dma_set_op(&dma_gemm_get, DMA_GET);
	dma_set_mode(&dma_gemm_get, PE_MODE);
	dma_set_reply(&dma_gemm_get, &replyget);	

	dma_set_op(&dma_gemm_put, DMA_PUT);
	dma_set_mode(&dma_gemm_put, PE_MODE);
	dma_set_reply(&dma_gemm_put, &replyput);
	int left_col_size,left_size;
  for(off = start; off < end; off ++)
  {
    m = off / num;
    n = off % num;
	  dma_set_size(&dma_gemm_get, col_size*sizeof(float));
	  dma_set_size(&dma_gemm_put, col_size*sizeof(float));
		for(i = 0;i+col_size-1 < N;i += col_size)
    {
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + n*M*N + m*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
    if(i < N){
      left_col_size = N % col_size;
	    dma_set_size(&dma_gemm_get, left_col_size*sizeof(float));
      
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
	    
      dma_set_size(&dma_gemm_put, left_col_size*sizeof(float));

		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + n*M*N + m*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
  }
  ldm_free(local_src, col_size*sizeof(float));
}
/*
void sw_slave_gemm_im2col_d(gemmTransPara *para) {
  const int  max_size = 48 * 1024;
  const int  SPNUM = 64; 
  int i,off,n,m;
  int id = athread_get_id(-1);
  int num = para->num;
  int M = para->M;
  int N = para->N;
  int count = M * num;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;
  if(local_count <1) return;
  int col_size = N;
  int max_rows_num = max_size / (col_size * sizeof(double));
  while(max_rows_num < 1)
  {
    col_size = col_size >>1;
    max_rows_num = max_size / (col_size * sizeof(double));
  }
  double * local_src = (double *)ldm_malloc(col_size*sizeof(double ));
  double * src_ptr = (double *)(para->src);
  double * dst_ptr = (double *)(para->dst);
  volatile int replyget=0,replyput=0;
  // DMA settings
	dma_set_op(&dma_gemm_get, DMA_GET);
	dma_set_mode(&dma_gemm_get, PE_MODE);
	dma_set_reply(&dma_gemm_get, &replyget);	
	//dma_set_size(&dma_gemm_get, col_size*sizeof(double));

	dma_set_op(&dma_gemm_put, DMA_PUT);
	dma_set_mode(&dma_gemm_put, PE_MODE);
	dma_set_reply(&dma_gemm_put, &replyput);
	//dma_set_size(&dma_gemm_put, col_size*sizeof(double));
	int left_col_size,left_size;
  for(off = start; off < end; off ++)
  {
    n = off / M;
    m = off % M;
	  dma_set_size(&dma_gemm_get, col_size*sizeof(double));
	  dma_set_size(&dma_gemm_put, col_size*sizeof(double));
		for(i = 0;i+col_size-1 < N;i += col_size)
    {
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + m*N*num + n*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
    if(i < N){
      left_col_size = N % col_size;
	    dma_set_size(&dma_gemm_get, left_col_size*sizeof(double));
      
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
	    
      dma_set_size(&dma_gemm_put, left_col_size*sizeof(double));

		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + m*N*num + n*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
  }
  ldm_free(local_src, col_size*sizeof(double));
}
void sw_slave_gemm_col2im_d(gemmTransPara *para) {
  const int  max_size = 48 * 1024;
  const int  SPNUM = 64; 
  int i,off,n,m;
  int id = athread_get_id(-1);
  int num = para->num;
  int M = para->M;
  int N = para->N;
  int count = M * num;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;
  if(local_count <1) return;
  int col_size = N;
  int max_rows_num = max_size / (col_size * sizeof(double));
  while(max_rows_num < 1)
  {
    col_size = col_size >>1;
    max_rows_num = max_size / (col_size * sizeof(double));
  }
  if(max_rows_num > local_count) max_rows_num = local_count;
  double * local_src = (double *)ldm_malloc(col_size*sizeof(double ));
  double * src_ptr = (double *)(para->src);
  double * dst_ptr = (double *)(para->dst);
  volatile int replyget=0,replyput=0;
  // DMA settings
	dma_set_op(&dma_gemm_get, DMA_GET);
	dma_set_mode(&dma_gemm_get, PE_MODE);
	dma_set_reply(&dma_gemm_get, &replyget);	

	dma_set_op(&dma_gemm_put, DMA_PUT);
	dma_set_mode(&dma_gemm_put, PE_MODE);
	dma_set_reply(&dma_gemm_put, &replyput);
	int left_col_size,left_size;
  for(off = start; off < end; off ++)
  {
    m = off / num;
    n = off % num;
	  dma_set_size(&dma_gemm_get, col_size*sizeof(double));
	  dma_set_size(&dma_gemm_put, col_size*sizeof(double));
		for(i = 0;i+col_size-1 < N;i += col_size)
    {
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + n*M*N + m*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
    if(i < N){
      left_col_size = N % col_size;
	    dma_set_size(&dma_gemm_get, left_col_size*sizeof(double));
      
      dma(dma_gemm_get,(long)(src_ptr + off*N + i),(long)(local_src));			
	    
      dma_set_size(&dma_gemm_put, left_col_size*sizeof(double));

		  dma_wait(&replyget,1);replyget=0;
      mb();
      dma(dma_gemm_put,(long)(dst_ptr + n*M*N + m*N + i),(long)(local_src));			
		  dma_wait(&replyput,1);replyput=0;
      mb();
    }
  }
  ldm_free(local_src, col_size*sizeof(double));
}*/
