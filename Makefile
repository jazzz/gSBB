CC = g++
GPU_CC = nvcc

FLAGS =  -g -O3 -Wall -Wextra -Wredundant-decls -Wno-sign-compare
DEP_FLAGS =  -c -g -O0 -Wall -Wextra -Wredundant-decls -Wno-sign-compare
GPU_FLAGS = -Xcompiler -rdynamic -lineinfo -g -G -c 
//GPU_FLAGS += -host-complilation 'C'

# GPU VARIABLES
CUDA_SDK = /home/jazz/NVIDIA_GPU_Computing_SDK
CUDA_SHARED = $(CUDA_SDK)/shared
CUDA_COMMON = $(CUDA_SDK)/C/common
THRUST = ./thrust/
INCLUDES = -I$(CUDA_COMMON)/inc -I$(CUDA_SHARED)/inc -I/usr/local/cuda/include/ -I$(THRUST)
GPU_LIBDIR = -L/usr/local/cuda/lib

LIBS = -lcudart -lpthread

HOST_SRC=$(wildcard *.cpp)
OBJS= EvoController.o Misc.o Main.o DataEngine.o Dataset.o

#all: scm scmgpu_pointstream scmgpu_pointstream_pivot scmgpu_chunk scmgpu_stream
#all: scmgpu_cpu scmgpu_simple scmgpu_chunk scmgpu_pivot scmgpu_selective scmgpu_best
all:  scmgpuNative scmgpuNativeWithTest scmgpuNativeCTSEL scmgpuNativeCPSEL scmgpuNativeCSEL

scmgpuNative:  EvoControllerLean.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o FreeMap.o
	$(GPU_CC) -o $@ $(LIBS)  EvoControllerLean.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o -lrt -lnvToolsExt

scmgpuNativeWithTest:  EvoControllerLeanWithTest.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o FreeMap.o
	$(GPU_CC) -o $@ $(LIBS)  EvoControllerLeanWithTest.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o -lrt -lnvToolsExt

scmgpuNativeCTSEL:  EvoControllerLeanCTSEL.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o FreeMap.o
	$(GPU_CC) -o $@ $(LIBS)  EvoControllerLeanCTSEL.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o -lrt -lnvToolsExt

scmgpuNativeCPSEL:  EvoControllerLeanCPSEL.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o FreeMap.o
	$(GPU_CC) -o $@ $(LIBS)  EvoControllerLeanCPSEL.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o -lrt -lnvToolsExt

scmgpuNativeCSEL:  EvoControllerLeanCSEL.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o FreeMap.o
	$(GPU_CC) -o $@ $(LIBS)  EvoControllerLeanCSEL.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o -lrt -lnvToolsExt

FreeMap.o: FreeMap.hpp FreeMap.cpp
	$(CC) $(DEP_FLAGS) FreeMap.hpp -o FreeMap.o

%.o : %.cpp
	 $(CC) $(DEP_FLAGS) $(INCLUDES) $< -o $@



EvoControllerLeanWithTest.o: EvoControllerLean.cpp EvoControllerLean.hpp
	 $(CC) $(DEP_FLAGS) $(INCLUDES) -DENABLE_TEST=1 EvoControllerLean.cpp -o EvoControllerLeanWithTest.o

EvoControllerLeanCPSEL.o: EvoControllerLean.cpp EvoControllerLean.hpp
	 $(CC) $(DEP_FLAGS) $(INCLUDES) -DCPSEL=1 EvoControllerLean.cpp -o EvoControllerLeanCPSEL.o

EvoControllerLeanCTSEL.o: EvoControllerLean.cpp EvoControllerLean.hpp
	 $(CC) $(DEP_FLAGS) $(INCLUDES) -DCTSEL=1 EvoControllerLean.cpp -o EvoControllerLeanCTSEL.o

EvoControllerLeanCSEL.o: EvoControllerLean.cpp EvoControllerLean.hpp
	 $(CC) $(DEP_FLAGS) $(INCLUDES) -DCPSEL=1 -DCTSEL=1 EvoControllerLean.cpp -o EvoControllerLeanCSEL.o

GpuTestController.o : GpuTestController.cu 
	$(GPU_CC) $(GPU_FLAGS) $(LIBS) $(INCLUDES) GpuTestController.cu

GpuController_Selection.o : GpuController_Selection.cu GpuController_Selection.cuh
	$(GPU_CC) $(GPU_FLAGS) $(LIBS) $(INCLUDES) GpuController_Selection.cu

GpuEval.o : GpuEval.cu GpuEval.cuh
	$(GPU_CC) $(GPU_FLAGS) $(LIBS) $(INCLUDES) GpuEval.cu

GpuMemController.o : GpuMemController.cu GpuMemController.cuh
	$(GPU_CC) $(GPU_FLAGS) $(LIBS) $(INCLUDES) GpuMemController.cu

CpuEval.o : CpuEval.cu
	$(GPU_CC) $(GPU_FLAGS) $(LIBS) $(INCLUDES) CpuEval.cu

CudaController.o:	CudaController.cu CudaControllerVars.cuh CudaControllerFunc.cuh
	$(GPU_CC) $(GPU_FLAGS) $(LIBS) $(INCLUDES) CudaController.cu

clean:
	rm *.o
