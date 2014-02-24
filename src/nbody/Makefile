CUDA_INSTALL_PATH ?= /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -lcuda

# Options
NVCCOPTIONS = -arch sm_20

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
GLFLAGS += -lm -lGL -lGLU -lglut 


.SUFFIXES:	.c	.cpp	.cu	.o	

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ 

all : 3dp 3d 2d

3dp: 3dprender.cpp.o kernel.cu.o kernel2.cu.o
	$(NVCC) -o 3dp 3dprender.cpp.o kernel.cu.o kernel2.cu.o $(LIB_CUDA) $(GLFLAGS)

3d: 3drender.cpp.o kernel.cu.o kernel2.cu.o
	$(NVCC) -o 3d 3drender.cpp.o kernel.cu.o kernel2.cu.o $(LIB_CUDA) $(GLFLAGS)
	
2d: 2drender.cpp.o kernel.cu.o kernel2.cu.o
	$(NVCC) -o 2d 2drender.cpp.o kernel.cu.o kernel2.cu.o $(LIB_CUDA) $(GLFLAGS)
	
clean:
	rm -rf *o *zip old/*o
