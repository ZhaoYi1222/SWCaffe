#include <slave.h>
#include <simd.h>
#include <dma.h>

#define SWAPABCD2(in0,in1,in2,in3){\
	floatv4 o0 = simd_vshff(in1,in0,68 );  \
	floatv4 o1 = simd_vshff(in1,in0,238);  \
	floatv4 o2 = simd_vshff(in3,in2,68 ); \
	floatv4 o3 = simd_vshff(in3,in2,238); \
	in0 = simd_vshff(o2,o0,136);  \
	in1 = simd_vshff(o2,o0,221);  \
	in2 = simd_vshff(o3,o1,136);  \
	in3 = simd_vshff(o3,o1,221);\
}
__thread_local dma_desc dma_trans_get,dma_trans_put;

typedef struct transPara_st {
  void *src;
  void *dst;
  int num,K,N;
}transPara;

inline int get_split_size(int size,int max_size)
{
	int val = size/max_size,split_size = 0;
	if(val<1) 
	{
      split_size = size - size%4;
	}
	else if(val>=max_size) split_size = max_size;
	else{
		int mod = size - size%4,tmp=0;

		split_size = 0;
		for(;val<max_size;val++)
		{
			tmp = mod/val;
			if(tmp <max_size && (tmp % 4 == 0))
			{
				split_size = tmp;
				break;
			}
		}
		if(split_size<16){
			split_size = (mod>>2);
			split_size = split_size - split_size%4;
		}
	}
	return split_size;
}
void sw_slave_trans_f(transPara *para) {
  const int  SIMDSIZE = 4;
  const int  SPNUM = 64; 
  const int  max_row = 24;
  const int  max_col = 128;
  int i,off,n,k,row,col,index;
  int id = athread_get_id(-1);
  int num = para->num;
  int K = para->K;
  int N = para->N;
  int count = K * num;
  int split_count = count / SIMDSIZE;
  int local_count = split_count/SPNUM + (id<(split_count%SPNUM));
  int start = id*(split_count/SPNUM) + id*(id<(split_count%SPNUM));
  local_count = local_count * SIMDSIZE;
  start = start * SIMDSIZE;
  int end = start + local_count;
  if(local_count <1) return;
  int split_row = get_split_size(local_count,max_row);
  int split_col = get_split_size(N,max_col);
  int size = split_row * split_col;
  float * local_src = (float *)ldm_malloc(size*sizeof(float ));
  float * local_dst = (float *)ldm_malloc(size*sizeof(float ));
  float * src_ptr = (float *)(para->src);
  float * dst_ptr = (float *)(para->dst);
  floatv4 va0,va1,va2,va3;
  volatile int replyget=0,replyput=0;
  // DMA settings
	dma_set_op(&dma_trans_get, DMA_GET);
	dma_set_mode(&dma_trans_get, PE_MODE);
	dma_set_reply(&dma_trans_get, &replyget);
	
	dma_set_op(&dma_trans_put, DMA_PUT);
	dma_set_mode(&dma_trans_put, PE_MODE);
	dma_set_reply(&dma_trans_put, &replyput);	
  int tmp_split_row,tmp_split_col;
  for(off = start; off+split_row-1 < end; off += split_row)
  {
    n = off / K;
    k = off % K;
    tmp_split_row = split_row;
    tmp_split_col = split_col;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(float));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(float));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(float));
	  dma_set_size(&dma_trans_put, size*sizeof(float));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(float));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(float));
		for(i = 0;i+tmp_split_col-1 < N;i += tmp_split_col)
		{
			dma(dma_trans_get,(long)(src_ptr + n*K*N + k*N + i),(long)(local_src));
			dma_wait(&replyget,1);replyget = 0;
			for(row = 0;row < tmp_split_row;row += SIMDSIZE)
			{
				for(col = 0;col < tmp_split_col;col += SIMDSIZE)
				{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
				}
			}
			dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K +k),(long)(local_dst));			
			dma_wait(&replyput,1);replyput=0;
		}	
    int left_num = (N - i) / SIMDSIZE;
    if(left_num < 1) continue;
		
    tmp_split_row = split_row;
    tmp_split_col = left_num*4;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(float));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(float));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(float));
	  dma_set_size(&dma_trans_put, size*sizeof(float));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(float));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(float));
		for(row = 0;row < tmp_split_row;row += SIMDSIZE)
		{
			for(col = 0;col < tmp_split_col;col += SIMDSIZE)
			{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
			 }
		 }
		 dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K +k),(long)(local_dst));			
		 dma_wait(&replyput,1);replyput=0;
  }

  if(off<local_count) {
    n = off / K;
    k = off % K;
    tmp_split_row = ((end - off)/SIMDSIZE)*SIMDSIZE;
    if(tmp_split_row >0){
    tmp_split_col = split_col;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(float));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(float));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(float));
	  dma_set_size(&dma_trans_put, size*sizeof(float));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(float));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(float));
		for(i = 0;i+tmp_split_col-1 < N;i += tmp_split_col)
		{
			dma(dma_trans_get,(long)(src_ptr + n*K*N + k*N + i),(long)(local_src));
			dma_wait(&replyget,1);replyget = 0;
			for(row = 0;row < tmp_split_row;row += SIMDSIZE)
			{
				for(col = 0;col < tmp_split_col;col += SIMDSIZE)
				{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
				}
			}
			dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K +k),(long)(local_dst));			
			dma_wait(&replyput,1);replyput=0;
		}	
    int left_num = (N -i) / SIMDSIZE;
    if(left_num >0) {
    tmp_split_row = split_row;
    tmp_split_col = left_num*SIMDSIZE;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(float));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(float));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(float));
	  dma_set_size(&dma_trans_put, size*sizeof(float));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(float));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(float));
    dma(dma_trans_get,(long)(src_ptr + n*K*N + k*N + i),(long)(local_src));
		dma_wait(&replyget,1);replyget = 0;
		for(row = 0;row < tmp_split_row;row += SIMDSIZE)
		{
			for(col = 0;col < tmp_split_col;col += SIMDSIZE)
			{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
			}
		}
		dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K + k),(long)(local_dst));			
		dma_wait(&replyput,1);replyput=0;
   }
   }
  }

  ldm_free(local_src, size*sizeof(float));
  ldm_free(local_dst, size*sizeof(float));
}
/*
void sw_slave_trans_d(transPara *para) {
  const int  SIMDSIZE = 4;
  const int  SPNUM = 64; 
  const int  max_row = 24;
  const int  max_col = 128;
  int i,off,n,k,row,col,index;
  int id = athread_get_id(-1);
  int num = para->num;
  int K = para->K;
  int N = para->N;
  int count = K * num;
  int split_count = count / SIMDSIZE;
  int local_count = split_count/SPNUM + (id<(split_count%SPNUM));
  int start = id*(split_count/SPNUM) + id*(id<(split_count%SPNUM));
  local_count = local_count * SIMDSIZE;
  start = start * SIMDSIZE;
  int end = start + local_count;
  if(local_count <1) return;
  int split_row = get_split_size(local_count,max_row);
  int split_col = get_split_size(N,max_col);
  int size = split_row * split_col;
  double * local_src = (double *)ldm_malloc(size*sizeof(double ));
  double * local_dst = (double *)ldm_malloc(size*sizeof(double ));
  double * src_ptr = (double *)(para->src);
  double * dst_ptr = (double *)(para->dst);
  doublev4 va0,va1,va2,va3;
  volatile int replyget=0,replyput=0;
  // DMA settings
	dma_set_op(&dma_trans_get, DMA_GET);
	dma_set_mode(&dma_trans_get, PE_MODE);
	dma_set_reply(&dma_trans_get, &replyget);
	
	dma_set_op(&dma_trans_put, DMA_PUT);
	dma_set_mode(&dma_trans_put, PE_MODE);
	dma_set_reply(&dma_trans_put, &replyput);	
  int tmp_split_row,tmp_split_col;
  for(off = start; off+split_row-1 < end; off += split_row)
  {
    n = off / K;
    k = off % K;
    tmp_split_row = split_row;
    tmp_split_col = split_col;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(double));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(double));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(double));
	  dma_set_size(&dma_trans_put, size*sizeof(double));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(double));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(double));
		for(i = 0;i+tmp_split_col-1 < N;i += tmp_split_col)
		{
			dma(dma_trans_get,(long)(src_ptr + n*K*N + k*N + i),(long)(local_src));
			dma_wait(&replyget,1);replyget = 0;
			for(row = 0;row < tmp_split_row;row += SIMDSIZE)
			{
				for(col = 0;col < tmp_split_col;col += SIMDSIZE)
				{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
				}
			}
			dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K +k),(long)(local_dst));			
			dma_wait(&replyput,1);replyput=0;
		}	
    int left_num = (N - i) / SIMDSIZE;
    if(left_num < 1) continue;
		
    tmp_split_row = split_row;
    tmp_split_col = left_num*4;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(double));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(double));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(double));
	  dma_set_size(&dma_trans_put, size*sizeof(double));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(double));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(double));
		for(row = 0;row < tmp_split_row;row += SIMDSIZE)
		{
			for(col = 0;col < tmp_split_col;col += SIMDSIZE)
			{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
			 }
		 }
		 dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K +k),(long)(local_dst));			
		 dma_wait(&replyput,1);replyput=0;
  }

  if(off<local_count) {
    n = off / K;
    k = off % K;
    tmp_split_row = ((end - off)/SIMDSIZE)*SIMDSIZE;
    if(tmp_split_row >0){
    tmp_split_col = split_col;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(double));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(double));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(double));
	  dma_set_size(&dma_trans_put, size*sizeof(double));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(double));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(double));
		for(i = 0;i+tmp_split_col-1 < N;i += tmp_split_col)
		{
			dma(dma_trans_get,(long)(src_ptr + n*K*N + k*N + i),(long)(local_src));
			dma_wait(&replyget,1);replyget = 0;
			for(row = 0;row < tmp_split_row;row += SIMDSIZE)
			{
				for(col = 0;col < tmp_split_col;col += SIMDSIZE)
				{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
				}
			}
			dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K +k),(long)(local_dst));			
			dma_wait(&replyput,1);replyput=0;
		}	
    int left_num = (N -i) / SIMDSIZE;
    if(left_num >0) {
    tmp_split_row = split_row;
    tmp_split_col = left_num*SIMDSIZE;
    size = tmp_split_row *tmp_split_col;
	  dma_set_size(&dma_trans_get, size*sizeof(double));
	  dma_set_bsize(&dma_trans_get, tmp_split_col*sizeof(double));
	  dma_set_stepsize(&dma_trans_get, (N - tmp_split_col)*sizeof(double));
	  dma_set_size(&dma_trans_put, size*sizeof(double));
	  dma_set_bsize(&dma_trans_put, tmp_split_row*sizeof(double));
	  dma_set_stepsize(&dma_trans_put, (K - tmp_split_row)*sizeof(double));
    dma(dma_trans_get,(long)(src_ptr + n*K*N + k*N + i),(long)(local_src));
		dma_wait(&replyget,1);replyget = 0;
		for(row = 0;row < tmp_split_row;row += SIMDSIZE)
		{
			for(col = 0;col < tmp_split_col;col += SIMDSIZE)
			{
					index = row*tmp_split_col+col;
					simd_load(va0,local_src+index);
					index = index+tmp_split_col;
					simd_load(va1,local_src+index);
					index = index+tmp_split_col;
					simd_load(va2,local_src+index);
					index = index+tmp_split_col;
					simd_load(va3,local_src+index);					
					
          index = col*tmp_split_row+row;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va1,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va2,local_dst+index);
					index = index+tmp_split_row;
					simd_store(va3,local_dst+index);				
			}
		}
		dma(dma_trans_put,(long)(dst_ptr + n*N*K + i*K + k),(long)(local_dst));			
		dma_wait(&replyput,1);replyput=0;
   }
   }
  }

  ldm_free(local_src, size*sizeof(float));
  ldm_free(local_dst, size*sizeof(float));
}
*/

