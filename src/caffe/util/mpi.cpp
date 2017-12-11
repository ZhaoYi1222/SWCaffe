#include "caffe/common.hpp"
#include "caffe/util/mpi.hpp"

#include <execinfo.h>
extern "C"{
#include "caffe/util/sw_dnn.h"
}
namespace caffe {

template<>
int caffe_mpi_send<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, MPI_FLOAT, dest, tag,
                    comm);
}

template<>
int caffe_mpi_send<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, MPI_DOUBLE, dest, tag,
                    comm);
}

int caffe_mpi_send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, datatype, dest, tag,
                    comm);
}
template<>
int caffe_mpi_recv<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, MPI_FLOAT, dest, tag,
                    comm, status);
}

template<>
int caffe_mpi_recv<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, MPI_DOUBLE, dest, tag,
                    comm, status);
}

int caffe_mpi_recv(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, datatype, dest, tag,
                    comm, status);
}

template <>
int caffe_mpi_isend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, MPI_FLOAT, dest, tag,comm, req);
}

template <>
int caffe_mpi_isend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag,comm, req);
}

int caffe_mpi_isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, datatype, dest, tag,comm, req);
}
template <>
int caffe_mpi_ssend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, MPI_FLOAT, dest, tag,comm);
}

template <>
int caffe_mpi_ssend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, MPI_DOUBLE, dest, tag,comm);
}

int caffe_mpi_ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, datatype, dest, tag,comm);
}

template <>
int caffe_mpi_iallreduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, MPI_Comm comm,MPI_Request*req  ){
  return MPI_Iallreduce(sendbuf, recvbuf, count, MPI_FLOAT, op, comm,req);
}

template <>
int caffe_mpi_iallreduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, MPI_Comm comm ,MPI_Request *req ){
  return MPI_Iallreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, comm,req);
}

template <>
int caffe_mpi_allreduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, MPI_Comm comm  ){
    //return MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, op, comm);
    int mpi_count,mpi_rank;
    int pof2 = 1,dest=0,simdsize=4;
    MPI_Comm_size(comm, &mpi_count);
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Request recv_req,send_req;
    MPI_Status  recv_statue;
    while(pof2 < mpi_count){
      dest = mpi_rank ^ pof2;
      if(dest < mpi_count){
	       MPI_Irecv(recvbuf, count,MPI_FLOAT, dest, pof2,comm, &recv_req);
	       MPI_Isend(sendbuf, count,MPI_FLOAT, dest, pof2,comm, &send_req);
	       //MPI_Send(sendbuf, count,MPI_FLOAT, dest, pof2,comm);
         MPI_Wait(&recv_req,&recv_statue);
         sw_add_f((float*)recvbuf,(float*)sendbuf,(float*)sendbuf,count);
      }
      pof2 = pof2 << 1;
    }
}

template <>
int caffe_mpi_allreduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, MPI_Comm comm  ){
  return MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
}

template <>
int caffe_mpi_reduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm  ){
    int comm_size,rank;
    MPI_Status status;
    MPI_Request send_req,recv_req;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    int mask = 0x1,source=0,tag = 10;
    int relrank = (rank - root + comm_size) % comm_size;

    sw_memcpy_f((float*)sendbuf,(float*)recvbuf,count);
    float * tmp_buff;
    tmp_buff = (float*)malloc(count * sizeof(float));
    while(mask < comm_size){
      // Receive
      if ((mask & relrank) == 0) {
        source = (relrank | mask);
        if (source < comm_size) {
          source = (source + root) % comm_size;
	        MPI_Irecv(tmp_buff,count,MPI_FLOAT,source,tag,comm,&recv_req);
          MPI_Wait(&recv_req,&status);
          sw_add_f((float*)tmp_buff,(float*)recvbuf,(float*)recvbuf,count);
        }
      }
      else {
         //I've received all that I'm going to.  Send my result to my parent 
         source = ((relrank & (~ mask)) + root) % comm_size;
	       MPI_Isend(recvbuf, count,MPI_FLOAT, source, tag,comm,&send_req);
         break;
      }
      mask = mask << 1;
    }
    free(tmp_buff);
    return 0;
  //return MPI_Reduce(sendbuf, recvbuf, count, MPI_FLOAT, op, root, comm);
}

template <>
int caffe_mpi_reduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm  ){
  return MPI_Reduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, root, comm);
}

template <>
int caffe_mpi_ireduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm, MPI_Request *req ){
  return MPI_Ireduce(sendbuf, recvbuf, count, MPI_FLOAT, op, root, comm, req);
}

template <>
int caffe_mpi_ireduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm, MPI_Request *req  ){
  return MPI_Ireduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, root, comm, req);
}

template <>
int caffe_mpi_bcast<float>( void *buffer, int count, int root,
    MPI_Comm comm ) {
  return MPI_Bcast(buffer, count, MPI_FLOAT, root, comm);
  
  int comm_size,rank;
  MPI_Status status;
  MPI_Request send_req,recv_req;
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &rank);
  int start = 0,mid = 0,tag = 1;
  int end = comm_size -1;
  while(1){
    if(start == end) break;
    mid = (start + end +1)>>1;
    if(rank >= start && rank <= mid -1)//front half
    {
      if(start == rank){
	      MPI_Isend(buffer, count,MPI_FLOAT,mid, tag,comm,&send_req);
      }
      end = mid - 1;
    }
    else if(rank >= mid && rank <= end){
      if(rank == mid){
        MPI_Irecv(buffer,count,MPI_FLOAT,start,tag,comm,&recv_req);
        MPI_Wait(&recv_req,&status);
      }
      start = mid;
    }
  }
}

template <>
int caffe_mpi_bcast<double>( void *buffer, int count, int root,
    MPI_Comm comm ) {
  return MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm);
}

}  // namespace caffe
