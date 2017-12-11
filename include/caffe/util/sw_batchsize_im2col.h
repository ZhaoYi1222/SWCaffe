/***************************************
 * Created by Liandeng Li
 * Date: 2017/11/8
 * Description: image to column functions
 *   accelerated in SW.
 **************************************/
#ifndef SW_BATCHSIZE_IM2COL_H_
#define SW_BATCHSIZE_IM2COL_H_
// data type: double
void swim2col_batchsize_d(const double* data_im, const int num,const int group,const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_col);
// data type: float
void swim2col_batchsize_f(const float* data_im, const int num,const int group,const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col);

// data type: double
void swcol2im_batchsize_d(const double* data_col,const int num, const int group,const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_im);

// data type: float
void swcol2im_batchsize_f(const float* data_col, const int num,const int group,const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im);


#endif
