#include <slave.h>
#include <simd.h>
#include <dma.h>
#include <assert.h>
#define min(a,b) ((a) < (b) ? (a) : (b))
__thread_local_fix dma_desc dma_get_A,dma_get_B,dma_put_C;

typedef struct biasPara_ {
  void *A,*B,*C;
  int num,M,N;
}biasPara;

inline void mb()
{
    asm volatile("":::"memory");
    asm volatile("memb");
}
void sw_slave_bias_f(biasPara *para) {
  const int  max_size = 48 * 1024;
  const int  SPNUM = 64; 
  const int  SIMDSIZE = 4;
  int i,off,j,k,m;
  int id = athread_get_id(-1);
  int M = para->M;
  int N = para->N;
  int num = para->num;
  int count = M;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int end = start + local_count;
  int col_size = N;
  int max_rows_num = max_size / (col_size * sizeof(float)*2);
  while(max_rows_num < 1)
  {
    col_size = col_size >>1;
    max_rows_num = max_size / (col_size * sizeof(float)*2);
  }
  if(id > M - 1)return;
  int max_m = 256;
  max_m = min(local_count,max_m);
  assert(col_size >1);
  assert(max_m>0);
  float * local_A = (float *)ldm_malloc(max_m*sizeof(float));
  float * local_B = (float *)ldm_malloc(col_size*sizeof(float));
  float * local_C = (float *)ldm_malloc(col_size*sizeof(float));
  float * A_ptr = (float *)(para->A);
  float * B_ptr = (float *)(para->B);
  float * C_ptr = (float *)(para->C);
  volatile int replyget_A=0,replyget_B=0,replyput_C=0;
  // DMA settings
	dma_set_op(&dma_get_A, DMA_GET);
	dma_set_mode(&dma_get_A, PE_MODE);
	dma_set_reply(&dma_get_A, &replyget_A);	
  dma_set_size(&dma_get_A,max_m*sizeof(float));

	dma_set_op(&dma_get_B, DMA_GET);
	dma_set_mode(&dma_get_B, PE_MODE);
	dma_set_reply(&dma_get_B, &replyget_B);	
	dma_set_size(&dma_get_B, col_size*sizeof(float));

	dma_set_op(&dma_put_C, DMA_PUT);
	dma_set_mode(&dma_put_C, PE_MODE);
	dma_set_reply(&dma_put_C, &replyput_C);
	dma_set_size(&dma_put_C, col_size*sizeof(float));
  floatv4 va,vb,vc;

  for(off = start; off + max_m - 1< end; off +=max_m)
  {
      dma(dma_get_A,(long)(A_ptr + off),(long)(local_A));			
		  dma_wait(&replyget_A,1);replyget_A=0;

      for(j=0;j<max_m;j++)
      {
          va = local_A[j];
	        dma_set_size(&dma_get_B, col_size*sizeof(float));
	        dma_set_size(&dma_put_C, col_size*sizeof(float));
	        for(i = 0;i+col_size-1 < N;i += col_size)
          {
             dma(dma_get_B,(long)(B_ptr + i),(long)(local_B));			
		         dma_wait(&replyget_B,1);replyget_B=0;
             for(k=0;k+SIMDSIZE-1<col_size;k+= SIMDSIZE)
             {
                simd_load(vb,local_B+k);
                vc = va*vb;
                simd_store(vc,local_B+k);
             }
             for(;k<col_size;k++)
             {
               local_B[k] = local_A[j]*local_B[k];
             }
             for(m=0;m<num;m++)
             {
                dma(dma_get_B,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyget_B,1);replyget_B=0;
                for(k=0;k+SIMDSIZE-1<col_size;k+= SIMDSIZE)
                {
                   simd_load(vb,local_B+k);
                   simd_load(vc,local_C+k);
                   vc = vb + vc;
                   simd_store(vc,local_C+k);
                }
                for(;k<col_size;k++)
                {
                  local_C[k] = local_B[k] + local_C[k];
                }
                dma(dma_put_C,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyput_C,1);replyput_C=0;
             }
          }
          if(i<N)
          {
            int left_col_size = N - i;
	          dma_set_size(&dma_get_B, left_col_size*sizeof(float));
	          dma_set_size(&dma_put_C, left_col_size*sizeof(float));
            dma(dma_get_B,(long)(B_ptr + i),(long)(local_B));			
		        dma_wait(&replyget_B,1);replyget_B=0;
            if(id<1)printf("left offset=%d\n",m*M*N+(off+j)*N+i);
            for(k=0;k+SIMDSIZE-1<left_col_size;k+= SIMDSIZE)
            {
                simd_load(vb,local_B+k);
                vc = va*vb;
                simd_store(vc,local_B+k);
            }
            for(;k<left_col_size;k++)
            {
               local_B[k] = local_A[j]*local_B[k];
            }
            for(m=0;m<num;m++)
            {
                dma(dma_get_B,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyget_B,1);replyget_B=0;
                for(k=0;k+SIMDSIZE-1<left_col_size;k+= SIMDSIZE)
                {
                   simd_load(vb,local_B+k);
                   simd_load(vc,local_C+k);
                   vc = vb + vc;
                   simd_store(vc,local_C+k);
                }
                for(;k<left_col_size;k++)
                {
                  local_C[k] = local_B[k] + local_C[k];
                }
                dma(dma_put_C,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyput_C,1);replyput_C=0;
            }
          }
      }
  }
  if(off < end)
  {
     dma_set_size(&dma_get_A, (end - off)*sizeof(float));
     dma(dma_get_A,(long)(A_ptr + off),(long)(local_A));			
		 dma_wait(&replyget_A,1);replyget_A=0;
     for(j=0;j<end - off;j++)
     {
          va = local_A[j];
	        dma_set_size(&dma_get_B, col_size*sizeof(float));
	        dma_set_size(&dma_put_C, col_size*sizeof(float));
	        for(i = 0;i+col_size-1 < N;i += col_size)
          {
             dma(dma_get_B,(long)(B_ptr + i),(long)(local_B));			
		         dma_wait(&replyget_B,1);replyget_B=0;
             for(k=0;k+SIMDSIZE-1<col_size;k+= SIMDSIZE)
             {
                simd_load(vb,local_B+k);
                vc = va*vb;
                simd_store(vc,local_B+k);
             }
             for(;k<col_size;k++)
             {
               local_B[k] = local_A[j]*local_B[k];
             }
             for(m=0;m<num;m++)
             {
                dma(dma_get_B,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyget_B,1);replyget_B=0;
                if(id<1)printf("main offset=%d\n",m*M*N+(off+j)*N+i);
                for(k=0;k+SIMDSIZE-1<col_size;k+= SIMDSIZE)
                {
                   simd_load(vb,local_B+k);
                   simd_load(vc,local_C+k);
                   vc = vb + vc;
                   simd_store(vc,local_C+k);
                }
                for(;k<col_size;k++)
                {
                  local_C[k] = local_B[k] + local_C[k];
                }
                dma(dma_put_C,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyput_C,1);replyput_C=0;
             }
          }
          if(i<N)
          {
            int left_col_size = N - i;
	          dma_set_size(&dma_get_B, left_col_size*sizeof(float));
	          dma_set_size(&dma_put_C, left_col_size*sizeof(float));
            dma(dma_get_B,(long)(B_ptr + i),(long)(local_B));			
		        dma_wait(&replyget_B,1);replyget_B=0;
            for(k=0;k+SIMDSIZE-1<left_col_size;k+= SIMDSIZE)
            {
                simd_load(vb,local_B+k);
                vc = va*vb;
                simd_store(vc,local_B+k);
            }
            for(;k<left_col_size;k++)
            {
               local_B[k] = local_A[j]*local_B[k];
            }
            for(m=0;m<num;m++)
            {
                dma(dma_get_B,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyget_B,1);replyget_B=0;
                if(id<1)printf("left offset=%d\n",m*M*N+(off+j)*N+i);
                for(k=0;k+SIMDSIZE-1<left_col_size;k+= SIMDSIZE)
                {
                   simd_load(vb,local_B+k);
                   simd_load(vc,local_C+k);
                   vc = vb + vc;
                   simd_store(vc,local_C+k);
                }
                for(;k<left_col_size;k++)
                {
                  local_C[k] = local_B[k] + local_C[k];
                }
                dma(dma_put_C,(long)(C_ptr + m*M*N+(off+j)*N+i),(long)(local_C));			
		            dma_wait(&replyput_C,1);replyput_C=0;
             }
            }
       }
  }
  ldm_free(local_A,max_m*sizeof(float));
  ldm_free(local_B,col_size*sizeof(float ));
  ldm_free(local_C,col_size*sizeof(float ));
}
