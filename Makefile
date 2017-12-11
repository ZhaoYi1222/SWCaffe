####COMPILERS####
LINK 	= 	mpiCC 
CXX 	=  	mpiCC -host -CG:pjump_all 
SWHCXX = sw5cc.new -host -msimd
SWSCXX = 	sw5cc.new -slave -CG:pjump_all -msimd
	
####FLAGS####
#basic compile flags
FLAGS = 	-O2 -OPT:IEEE_arith=2 -OPT:Olimit=0 
#caffe compile flags
FLAGS += 	-DCPU_ONLY
FLAGS +=  -DUSE_OPENCV
FLAGS +=  -DUSE_LMDB
#swcaffe compile flags
FLAGS += 	-DUSE_MPI
#FLAGS +=  -DUSE_4CG
#swdnn flags
#FLAGS += -DUSE_SWDNN
#FLAGS += -DSW_TRANS

FLAGS += -DUSE_CONV
#FLAGS += -DCHECK_CONV
#FLAGS += -DCHECK_DROPOUT
#FLAGS += -DUSE_SWBASE
#FLAGS += -DDEBUG_SWBASE
FLAGS += -DUSE_SWPOOL
FLAGS += -DUSE_SWRELU
FLAGS += -DUSE_SWIM2COL
#FLAGS += -DUSE_SWPRELU
FLAGS += -DUSE_SWSOFTMAX
#FLAGS += -DDEBUG_PRINT_TIME
#debug flags
#alogrithm logic and forbackward time
FLAGS += 	-DDEBUG_VERBOSE_1
#time of each layer
FLAGS += 	-DDEBUG_VERBOSE_2
#FLAGS += 	-DDEBUG_VERBOSE_3
#print timer in sw_conv_layer_impl
#FLAGS += 	-DDEBUG_VERBOSE_3
#address and length of mpibuff
#FLAGS += 	-DDEBUG_VERBOSE_6
#in sgd solvers data value print
#FLAGS +=  -DDEBUG_VERBOSE_7
#ld flags
LDFLAGS = -lm_slave -lm
LDFLAGS += -allshare
#include flags
SWINC_FLAGS=-I./include -I$(THIRD_PARTY_DIR)/include

####DIRS####
SRC = ./src
SWBUILD_DIR=./swbuild
THIRD_PARTY_DIR=../thirdparty
BIN_DIR=./bin

####SRC####
caffesrc=$(wildcard ./src/caffe/*.cpp ./src/caffe/layers/*.cpp ./src/caffe/solvers/*.cpp ./src/caffe/util/*.cpp ./src/glog/*.cpp)
caffepbsrc = ./src/caffe/proto/caffe.pb.cc
swhostsrc = $(wildcard ./src/caffe/swutil/*.c)
swslavesrc = $(wildcard ./src/caffe/swutil/slave/*.c)
swslavessrc = $(wildcard ./src/caffe/swutil/slave/*.S)

####OBJS####
caffeobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.cpp, %.o, $(caffesrc)))
caffepbobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.cc, %.o, $(caffepbsrc)))
swhostobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.c, %.o, $(swhostsrc)))
swslaveobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.c, %.o, $(swslavesrc)))
swslavesobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.S, %_asm.o, $(swslavessrc)))
allobjs = $(caffeobjs) $(caffepbobjs) $(swhostobjs) $(swslaveobjs) $(swslavesobjs)

#libraries
SWLIBOBJ=$(THIRD_PARTY_DIR)/lib/cblas_LINUX0324.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libswblasall-2.a
#SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libswblas0324.a
#SWLIBOBJ+=-Wl,--whole-archive $(THIRD_PARTY_DIR)/lib/libhdf5.a
#SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libhdf5_hl.a -Wl,--no-whole-archive
SWLIBOBJ+=-Wl,--whole-archive $(THIRD_PARTY_DIR)/lib/libopencv_core.a -Wl,--no-whole-archive
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libopencv_highgui.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libopencv_imgproc.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libjpeg.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libz.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libprotobuf.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libboost_system.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libboost_thread.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libboost_atomic.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libgflags.a
#######order matters
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/liblmdb.a

####Rules####
#debug makefile
show:
	echo $(caffesrc)
	echo $(caffepbsrc)
	echo $(swhostsrc)
	echo $(swslavesrc)
	echo $(swslavessrc)
	echo $(caffeobjs)
	echo $(caffepbobjs)
	echo $(swhostobjs)
	echo $(swslaveobjs)
	echo $(swslavesobjs)

mpi_caffe: $(BIN_DIR)/mpi_caffe_sw
caffe: $(BIN_DIR)/caffe_sw
convert_imageset: $(BIN_DIR)/convert_imageset_sw
compute_image_mean: $(BIN_DIR)/compute_image_mean_sw
convert_cifar: $(BIN_DIR)/convert_cifar_sw
mk:
	mkdir -p $(SWBUILD_DIR) $(SWBUILD_DIR)/caffe $(SWBUILD_DIR)/caffe/util $(SWBUILD_DIR)/caffe/layers \
		$(SWBUILD_DIR)/caffe/swutil $(SWBUILD_DIR)/caffe/swutil/slave $(SWBUILD_DIR)/caffe/proto\
		$(SWBUILD_DIR)/caffe/solvers $(SWBUILD_DIR)/glog  $(BIN_DIR)

#caffe tools
$(BIN_DIR)/convert_cifar_sw: $(SWBUILD_DIR)/convert_cifar_sw.o $(allobjs)
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/convert_cifar_sw.o: ./tools/convert_cifar_data.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(BIN_DIR)/compute_image_mean_sw: $(SWBUILD_DIR)/compute_image_mean_sw.o $(allobjs)
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/compute_image_mean_sw.o: ./tools/compute_image_mean.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(BIN_DIR)/convert_imageset_sw: $(SWBUILD_DIR)/convert_imageset_sw.o $(allobjs)
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/convert_imageset_sw.o: ./tools/convert_imageset.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/caffe_sw:$(allobjs) $(SWBUILD_DIR)/caffe_sw.o
	$(LINK) $^ $(LDFLAGS) -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/caffe_sw.o: ./tools/caffe.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@


$(BIN_DIR)/mpi_caffe_sw:$(allobjs) $(SWBUILD_DIR)/mpi_caffe_sw.o
	$(LINK) $^ $(LDFLAGS) -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/mpi_caffe_sw.o: ./tools/mpi_caffe.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/swutil/slave/gemm_asm.o: ./src/caffe/swutil/slave/gemm.S
	$(SWSCXX) $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
$(SWBUILD_DIR)/caffe/swutil/slave/gemm_float_asm.o: ./src/caffe/swutil/slave/gemm_float.S
	$(SWSCXX) $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
$(SWBUILD_DIR)/caffe/swutil/slave/%.o: ./src/caffe/swutil/slave/%.c
	$(SWSCXX) -c $< $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swutil/%.o: ./src/caffe/swutil/%.c
	$(SWHCXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/%.o: ./src/caffe/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/layers/%.o: ./src/caffe/layers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/solvers/%.o: ./src/caffe/solvers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/%.o: ./src/caffe/util/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/glog/%.o: ./src/glog/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/proto/%.o: ./src/caffe/proto/%.cc
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

clean:
	rm -f $(allobjs) $(SWBUILD_DIR)/*.o core* $(BIN_DIR)/*
