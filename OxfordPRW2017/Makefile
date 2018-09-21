
#INC	:= -I$(CUDA_HOME)/include -I.
#LIB	:= -L$(CUDA_HOME)/lib64 -lcudart

#replace the 'sm_50' value in the line below with the name of the class of
#NVIDIA GPU architecture for which the CUDA input files must be compiled
NVCCFLAGS	:= -lineinfo -arch=sm_50 --ptxas-options=-v --use_fast_math

all:		prw_sync 2Dprw_bal 3Dprw_bal

prw_sync:	prw_sync.cu Makefile
		nvcc prw_sync.cu -o prw_sync $(NVCCFLAGS)

2Dprw_bal:	2Dprw_bal.cu Makefile
		nvcc 2Dprw_bal.cu -o 2Dprw_bal $(NVCCFLAGS)

3Dprw_bal:	3Dprw_bal.cu Makefile
		nvcc 3Dprw_bal.cu -o 3Dprw_bal $(NVCCFLAGS)

clean:
		rm prw_sync 2Dprw_bal 3Dprw_bal
