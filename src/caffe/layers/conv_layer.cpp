#include <vector>
#include <assert.h>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#define USE_SWDNN
//#define TEST
//#ifdef USE_SWDNN

extern "C" {
#include "caffe/swlayers/sw_conv_layer_impl.h"
}
//#endif

static int times = 0;
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifdef USE_SWDNN
  //assert(typeid(Dtype) == typeid(double));
  const Dtype* weight       = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const int* stride_data = this->stride_.cpu_data();
    const int* pad_data = this->pad_.cpu_data();
    int mypad = 0;
    if(this->num_spatial_axes_)
      mypad = pad_data[0];

    const int* dilation_data = this->dilation_.cpu_data();
    if(bottom[0]->num() >= 128 
        && bottom[0]->channels() >= 64 && bottom[0]->channels() % 32 == 0 
        && top[0]->channels() >= 64 && top[0]->channels() % 32 == 0
        && this->group_==1
      ){
      const Dtype* bottom_data  = bottom[i]->cpu_data();
      Dtype* top_data           = top[i]->mutable_cpu_data();

      if(sizeof(Dtype) == sizeof(double))
      {
        //sw_conv_forward_impl_d(
        sw_conv_forward_pad_impl_d(
            (double*)bottom_data,
            (double*)weight,
            (double*)top_data,
            //bias_data,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            //int pad
            mypad
            );
      }
      else 
      {

        sw_conv_forward_pad_impl_f(
            (float*)bottom_data,
            (float*)weight,
            (float*)top_data,
            //bias_data,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            //int pad
            mypad
            );
      }
    }

    else {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data
            + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
      }
    }
#ifdef USE_CONV
      if (this->bias_term_) {
          Dtype* top_data = top[i]->mutable_cpu_data();
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_batchsize_cpu_bias(this->num_,top_data, bias);
      }
#else
    if (this->bias_term_) {
      Dtype* top_data = top[i]->mutable_cpu_data();
      const Dtype* bias = this->blobs_[1]->cpu_data();
      for (int n = 0; n < this->num_; ++n)
        this->forward_cpu_bias(top_data
            + n * this->top_dim_, bias);
    }
#endif
  }//for
#else
#ifdef CHECK_CONV
   Blob<Dtype> my_bottom_blob;
   Blob<Dtype> my_top_blob;
   const Dtype * mybottom_data;
   Dtype * mytop_data;
    my_bottom_blob.CopyFrom(*bottom[0], false, true);
    my_top_blob.CopyFrom(*top[0],false , true);
    mybottom_data  = my_bottom_blob.cpu_data();
    mytop_data     = my_top_blob.mutable_cpu_data();
    const Dtype* myweight = this->blobs_[0]->cpu_data();
    const Dtype* mybias = this->blobs_[1]->cpu_data();
    for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(mybottom_data + n * this->bottom_dim_, myweight,
            mytop_data + n * this->top_dim_);
        if (this->bias_term_) {
          this->forward_cpu_bias(mytop_data + n * this->top_dim_, mybias);
        }
    }
#endif
#ifdef DEBUG_VERBOSE_3
    this->im2col_time = 0;
    this->forward_cpu_gemm_time = 0;
    this->forward_cpu_bias_time = 0;
#endif
#ifdef USE_CONV
   const Dtype* weight = this->blobs_[0]->cpu_data();
   for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) 
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
      //this->forward_batchsize_cpu_gemm(this->num_,this->bottom_dim_,bottom_data, weight,top_data);
      if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_batchsize_cpu_bias(this->num_,top_data, bias);
      }
    }
#else
   const Dtype* weight = this->blobs_[0]->cpu_data();
   for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->cpu_data();
      Dtype* top_data = top[i]->mutable_cpu_data();
      for (int n = 0; n < this->num_; ++n) {
        this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    }
#endif
#ifdef DEBUG_VERBOSE_3
    printf("forward_cpu_gemm= %lf forward_cpu_bias_time= %lf im2col_time = %lf\n",this->forward_cpu_gemm_time,this->forward_cpu_bias_time,this->im2col_time);
#endif
#ifdef CHECK_CONV
    double sum=0.0, sum_ref=0.0;
    int times = 0;
    Dtype *top_data = top[0]->mutable_cpu_data();
    mytop_data     = my_top_blob.mutable_cpu_data();
    for(int ii = 0; ii < top[0]->count(); ++ii){
       if(fabs(mytop_data[ii] - top_data[ii])>1e-3) 
       {
         if(times++ <10) 
         printf("count=%d i= %d %f vs %f\n",top[0]->count(),ii,mytop_data[ii],top_data[ii]);
       }
        sum     += mytop_data[ii];
        sum_ref += top_data[ii];
    }
    DLOG(INFO) << "FORWARD CONV DATA: " << sum << " SUM_REF : " << sum_ref ;
#endif
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#ifdef CHECK_CONV
    Blob<Dtype> my_bottom_blob;
    Blob<Dtype> my_top_blob;
    Blob<Dtype> my_weight_blob;
    Blob<Dtype> my_bias_blob;

      my_bottom_blob.CopyFrom(*bottom[0], true, true);
      my_bottom_blob.CopyFrom(*bottom[0], false, true);
      my_top_blob.CopyFrom(*top[0], true, true);

      my_weight_blob.CopyFrom(*(this->blobs_[0]), false, true);
      my_weight_blob.CopyFrom(*(this->blobs_[0]), true, true);
      my_bias_blob.CopyFrom(*(this->blobs_[1]), true, true);

      const Dtype* myweight       = my_weight_blob.cpu_data();
      Dtype* myweight_diff        = my_weight_blob.mutable_cpu_diff();
      const Dtype* mybottom_data  = my_bottom_blob.cpu_data();
      Dtype* mybottom_diff        = my_bottom_blob.mutable_cpu_diff();
      Dtype* mytop_diff           = my_top_blob.mutable_cpu_diff();
      Dtype* mybias_diff          = my_bias_blob.mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(mybias_diff, mytop_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[0]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(mybottom_data + n * this->bottom_dim_,
              mytop_diff + n * this->top_dim_, myweight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[0]) {
          this->backward_cpu_gemm(mytop_diff + n * this->top_dim_, myweight,
              mybottom_diff + n * this->bottom_dim_);
        }
      }
    }
#endif

#ifdef USE_SWDNN
    //assert(typeid(Dtype) == typeid(double));
    const Dtype* weight    = this->blobs_[0]->cpu_data();
    Dtype* weight_diff     = this->blobs_[0]->mutable_cpu_diff();

      int mypad = 0;
      const int* pad_data = this->pad_.cpu_data();
      if(this->num_spatial_axes_)
        mypad = pad_data[0];

    for (int i = 0; i < top.size(); ++i) {
      const Dtype* bottom_data  = bottom[i]->cpu_data();
      Dtype* bottom_diff        = bottom[i]->mutable_cpu_diff();
      const Dtype* top_diff     = top[i]->mutable_cpu_diff();
#ifdef USE_CONV
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      this->backward_batchsize_cpu_bias(this->num_,bias_diff, top_diff);
    }
#else
      if (this->bias_term_ && this->param_propagate_down_[1]) {
          Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
          for (int n = 0; n < this->num_; ++n) {
            this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
          }
      }
#endif

    if(    bottom[0]->num() >= 64 && bottom[0]->num() % 32 == 0 
        && bottom[0]->channels() >= 128 && bottom[0]->channels()%128 == 0
        && top[0]->channels() >= 64 && top[0]->channels() % 32 == 0 
        && this->group_==1){
//LOG(INFO)<<"Backward_cpu USE SWDNN bottom channels: " <<bottom[0]->channels() << " top channels: "<<top[0]->channels();
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        if(sizeof(Dtype)== sizeof(double))
		{
          sw_conv_backward_pad_impl_d(
            //const Type* in,
            (double*)bottom_data,
            //const Type* out_grad,
            (double*)top_diff,
            //Type* weight,
            (double*)weight,
            //Type* in_grad,
            (double*)bottom_diff,
            //Type* weight_diff,
            (double*)weight_diff,
            //Type* bias_grad,
            //bias_diff,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            mypad
            );
		}
		else 
		{
          sw_conv_backward_pad_impl_f(
            //const Type* in,
            (float*)bottom_data,
            //const Type* out_grad,
            (float*)top_diff,
            //Type* weight,
            (float*)weight,
            //Type* in_grad,
            (float*)bottom_diff,
            //Type* weight_diff,
            (float*)weight_diff,
            //Type* bias_grad,
            //bias_diff,
            //int Ci,
            bottom[0]->width(),
            //int Ri,
            bottom[0]->height(),
            //int K,
            this->kernel_shape_.cpu_data()[0],
            //int Ni,
            bottom[0]->channels(),
            //int No,
            top[0]->channels(),
            //int B
            bottom[0]->num(),
            mypad
            );
		}
        }
    }
    else {
      if (this->param_propagate_down_[0] || propagate_down[i]) {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }//else
  }//for
#else
#ifdef DEBUG_VERBOSE_3
    this->im2col_time = 0;
    this->col2im_time = 0;
    this->backward_cpu_gemm_time = 0;
    this->backward_cpu_bias_time = 0;
    this->weight_cpu_gemm_time = 0;
#endif
#ifdef USE_CONV
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      this->backward_batchsize_cpu_bias(this->num_,bias_diff, top_diff);
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
    /*
    if (this->param_propagate_down_[0] || propagate_down[i]) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->batchsize_weight_cpu_gemm(this->num_,this->bottom_dim_,bottom_data,top_diff, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_batchsize_cpu_gemm(this->num_,this->bottom_dim_,top_diff, weight,bottom_diff);
        }
    }*/
  }
#else
  for (int i = 0; i < top.size(); ++i) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
#endif
#ifdef DEBUG_VERBOSE_3
    printf("backward_cpu_gemm_time= %lf backward_cpu_bias_time= %lf weight_cpu_gemm_time= %lf im2col_time= %lf col2im_time = %lf\n",this->backward_cpu_gemm_time,this->backward_cpu_bias_time,this->weight_cpu_gemm_time,this->im2col_time,this->col2im_time);
#endif
#endif
#ifdef CHECK_CONV
  times = 0;
  double dSum1=0,dSum2=0;
  assert( bottom[0]->count() == my_bottom_blob.count() );
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
   for(int i = 0; i < bottom[0]->count(); ++i){
    if( fabs(bottom_diff[i] - mybottom_diff[i]) > 1e-4)
      if(times ++ <10)
       printf("old= %5.3f my=%5.3f\n",bottom_diff[i], mybottom_diff[i]);
    dSum1 += bottom_diff[i];
    dSum2 += mybottom_diff[i];
  }
  printf("CHECK BACK bottom diff dSum1=%5.3f dSum2=%5.3f\n",dSum1,dSum2);
 
  assert( this->blobs_[0]->count() == my_weight_blob.count() );
  times = 0;
  dSum1=0;
  dSum2=0;
  weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for(int i = 0; i < this->blobs_[0]->count(); ++i){
    if( fabs(weight_diff[i] - myweight_diff[i]) > 1e-4)
      if(times ++ <10)
       printf("old= %5.3f my=%5.3f\n",weight_diff[i], myweight_diff[i]);
    dSum1 += weight_diff[i];
    dSum2 += myweight_diff[i];
  }
  printf("CHECK BACK weight diff dSum1=%5.3f dSum2=%5.3f\n",dSum1,dSum2);
  
  times = 0;
  dSum1=0;
  dSum2=0;
  Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
  for(int i = 0; i < this->blobs_[1]->count(); ++i){
    if( fabs(bias_diff[i] - mybias_diff[i]) > 1e-4)
      if(times ++ <10)
      printf("old=%5.3f my=%5.3f\n",bias_diff[i],mybias_diff[i]);
    dSum1 += bias_diff[i];
    dSum2 += mybias_diff[i];
  }
  printf("CHECK BACK bias diff dSum1=%5.3f dSum2=%5.3f\n",dSum1,dSum2);
#endif
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
