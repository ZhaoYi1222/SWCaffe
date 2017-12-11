// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"
#ifdef USE_SWBASE
extern "C"{
#include "caffe/util/sw_dnn.h"
}
#endif
namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
#ifdef USE_SWBASE1
    if(typeid(float) == typeid(Dtype))
       sw_dropout_layer_f((float*)bottom_data,(unsigned int*)mask,(float*)top_data,(float)scale_,count);
    else
       sw_dropout_layer_d((double*)bottom_data,(unsigned int*)mask,(double*)top_data,(double)scale_,count);

#ifdef CHECK_DROPOUT
    Dtype * p_data = (Dtype*)malloc(count*sizeof(Dtype));
    for (int i = 0; i < count; ++i) {
      p_data[i] = bottom_data[i] * mask[i] * scale_;
    }
    
    double dSum1=0,dSum2=0;
    int times = 0;
    top_data = top[0]->mutable_cpu_data();
    for(int i=0;i<count;i++){
        if(fabs(top_data[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",top_data[i],p_data[i]);
        }
        dSum1 += top_data[i];
        dSum2 += p_data[i];
    }
    printf("DropoutLayer Forward_cpu dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
    free(p_data);
#endif
#else
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
#endif
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
#ifdef USE_SWBASE1
    if(typeid(float) == typeid(Dtype))
       sw_dropout_layer_f((float*)top_diff,(unsigned int*)mask,(float*)bottom_diff,(float)scale_,count);
    else
       sw_dropout_layer_d((double*)top_diff,(unsigned int*)mask,(double*)bottom_diff,(double)scale_,count);
#ifdef CHECK_DROPOUT
    Dtype * p_data = (Dtype*)malloc(count*sizeof(Dtype));
    for (int i = 0; i < count; ++i) {
       p_data[i] = top_diff[i] * mask[i] * scale_;
    }
    
    double dSum1=0,dSum2=0;
    int times = 0;
    bottom_diff = bottom[0]->mutable_cpu_diff();
    for(int i=0;i<count;i++){
        if(fabs(bottom_diff[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",bottom_diff[i],p_data[i]);
        }
        dSum1 += bottom_diff[i];
        dSum2 += p_data[i];
    }
    printf("DropoutLayer Backward_cpu dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
    free(p_data);
#endif
#else
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * mask[i] * scale_;
      }
#endif
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DropoutLayer);
#endif

INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
