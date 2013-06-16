CC := g++
GPU_CC := nvcc
LD        := nvcc

FLAGS =  -g -O3 -Wall -Wextra -Wredundant-decls -Wno-sign-compare
GPU_FLAGS = -Xcompiler -rdynamic -lineinfo -g -G -c 

# GPU VARIABLES
CUDA_SDK = /home/jazz/NVIDIA_GPU_Computing_SDK
CUDA_SHARED = $(CUDA_SDK)/shared
CUDA_COMMON = $(CUDA_SDK)/C/common
THRUST = ./thrust/

MODULES			:= core dataset utils
SRC_DIR			:= $(addprefix src/,$(MODULES))
BUILD_DIR		:= $(addprefix build/,$(MODULES))

SRC			:= $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ			:= $(patsubst src/%.cpp,build/%.o,$(SRC))

CUDA_MODULES	:= cuda
CUDA_SRC_DIR	:= $(addprefix src/,$(CUDA_MODULES))
CUDA_BUILD_DIR	:= $(addprefix build/,$(CUDA_MODULES))

CUDA_SRC		:= $(foreach sdir,$(CUDA_SRC_DIR),$(wildcard $(sdir)/*.cu))
CUDA_OBJ		:= $(patsubst src/%.cu,build/%.o,$(CUDA_SRC))

INCLUDES	:= $(addprefix -I,$(SRC_DIR))
INCLUDES	+= $(addprefix -I,$(CUDA_SRC_DIR))

INCLUDES	+= -I$(CUDA_COMMON)/inc -I$(CUDA_SHARED)/inc -I/usr/local/cuda/include/ -I$(THRUST)


vpath %.cpp $(SRC_DIR)
vpath %.cu $(CUDA_SRC_DIR)
 

define make-cuda-goal
$1/%.o: %.cu
	$(GPU_CC) $(GPU_FLAGS) $(INCLUDES) $$< -c -o $$@
endef
       
define make-goal
$1/%.o: %.cpp
	echo "MAKEGOALS"
	$(CC) $(FLAGS) $(INCLUDES) $$< -c -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs bin/scmGP

bin/scmGP: $(OBJ) $(CUDA_OBJ) build/FreeMap.o
	$(LD) $(INCLUDES) $^ -o $@ -lrt -lnvToolsExt -lcudart -lpthread

bin/scmgpuNative:  EvoControllerLean.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o FreeMap.o
	$(LD) -o $@ $(LIBS)  EvoControllerLean.o Misc.o Main.o DataEngine.o Dataset.o GpuTestController.o CudaController.o -lrt -lnvToolsExt

checkdirs: $(BUILD_DIR) $(CUDA_BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

$(CUDA_BUILD_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD_DIR) $(CUDA_BUILD_DIR)

build/FreeMap.o: src/FreeMap.hpp src/FreeMap.cpp
	$(CC) -c -g -O3 -Wall -Wextra -Wredundant-decls -Wno-sign-compare src/FreeMap.cpp -o build/FreeMap.o


$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))
$(foreach bdir,$(CUDA_BUILD_DIR),$(eval $(call make-cuda-goal,$(bdir))))
