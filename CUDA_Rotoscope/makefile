####################################################################
#  Makefile for Rotoscope project. (OpenCV 3 version)
#
#  OpenCV installed on Ubuntu 16.04 using:
#  http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/
#
#  When building OpenCV3 code use cmake flags
#  WITH_CUDA=ON
#  ENABLE_FAST_MATH=1
#  CUDA_FAST_MATH=1
#  WITH_CUBLAS=1
#
#  Created by Michael Davis 7/17/2018
#
###################################################################

TARGET = CUDA_Rotoscope.exe
LIBS = `pkg-config --cflags --libs opencv4` -I"cub-1.8.0/"
CC = nvcc -lineinfo
CFLAGS = -std=c++11 -Xcompiler -fopenmp -g


.PHONY: default all clean

default: $(TARGET)
all: default


display: $(TARGET)


OBJECTS = main.cpp Roto.cu


#%.o: %.c $(HEADERS)
#	$(CC) $(CFLAGS) -c $< -o $@

#.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(PFLAGS) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *_roto.avi
	-rm -f $(TARGET)
