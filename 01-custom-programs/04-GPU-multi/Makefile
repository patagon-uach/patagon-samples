##############################################################
# Makefile                                                   #
#                                                            #
# Author      : Cristobal Navarro <crinavar@dcc.uchile.cl>   #
# Version     : 1.1                                          #
# Date        : April 2021                                   #
# Description : Compiles the program                         #
##############################################################

# hierarchy, names
BIN		:= ./bin
OBJ		:= ./obj
SRC		:= ./src
DEP		:= ./dep
EXEC	:= trueke
CXXFLAGS   := -O3 -g -fopenmp

# [CAN MODIFY] set these paths according to your system
CUDA_ROOT ?= /usr/local/cuda
CUDA_INSTALL_PATH ?= $(CUDA_ROOT)
CUDA_SAMPLES_PATH ?= $(CUDA_ROOT)/samples
CUDA_COMMON_PATH ?= $(CUDA_SAMPLES_PATH)/common

# [CAN MODIFY] compiler, include and lib parameters
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_COMMON_PATH)/inc" -I"$(CUDA_INSTALL_PATH)/include"
LIBS = -lcuda -L"$(CUDA_INSTALL_PATH)/lib64" -lcudart -L"$(CUDA_COMMON_PATH)/lib" -lpthread -lnvidia-ml -lgomp

# [CAN MODIFY] compiler flags, update them if they look too old for your hardware
SPECIAL_FLAGS = --compiler-options -fno-strict-aliasing,-O3,-march=native,-march=znver2,-mavx2,-funroll-loops,-finline-functions -Xcompiler -fopenmp -Xcompiler
NVCCFLAGS	:= -m64 -arch sm_80 -lineinfo -O3 -Xptxas -dlcm=cg ${SPECIAL_FLAGS} -D_FORCE_INLINES
DEBUG_NVCCFLAGS := --ptxas-options=-v -G -g-ccbin /usr/bin/g++ -Xptxas -dlcm=cg

# source files
CPP_SOURCES	:= $(wildcard $(SRC)/*.cpp)
CPP_HEADERS	:= $(wildcard $(SRC)/*.h)
CU_SOURCES	:= $(wildcard $(SRC)/*.cu)
CU_HEADERS	:= $(wildcard $(SRC)/*.cuh)

# object files
CPP_OBJS	:= $(patsubst $(SRC)/%.cpp, $(OBJ)/%.o, $(CPP_SOURCES))
CU_OBJS		:= $(patsubst $(SRC)/%.cu, $(OBJ)/%.cuo, $(CU_SOURCES))

# dependencies
CPP_DEPS	:= $(patsubst $(OBJ)/%.o, $(DEP)/%.d,$(CPP_OBJS))
CU_DEPS		:= $(patsubst $(OBJ)/%.cuo, $(DEP)/%.cud,$(CU_OBJS))

all: $(EXEC)


$(OBJ)/%.cuo : $(SRC)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<
	$(NVCC) $(NVCCFLAGS) $(INCD) -M $(SRC)/$*.cu -MT $(OBJ)/$*.cuo > $(DEP)/$*.cud

$(OBJ)/%.o: $(SRC)/%.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<
	$(CXX) -M $(SRC)/$*.cpp -MT $(OBJ)/$*.o > $(DEP)/$*.d

# rules 
$(EXEC): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN)/$(EXEC) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN)/$(EXEC) $(OBJ)/*.o $(OBJ)/*.cuo $(DEP)/*.cud $(DEP)/*.d

# include dependency files
-include $(CU_DEPS)
-include $(CPP_DEPS)
