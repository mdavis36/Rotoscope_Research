####################################################################
#  Makefile for Rotoscope project. (OpenCV 3 version)
#
#  OpenCV installed on Ubuntu 16.04 using:
#  http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/
#
#  Created by Michael Davis 7/17/2018
#
###################################################################

TARGET = Rotoscope.exe
LIBS = `pkg-config --cflags --libs opencv`
CC = g++
CFLAGS = -std=c++11 -fopenmp


.PHONY: default all clean

default: $(TARGET)
all: default

display: CC += -D_DEBUG -D_DISPLAY_ALL
debug: CC += -D_DEBUG

display: $(TARGET)
debug: $(TARGET)


OBJECTS = main.cpp videoIO.cpp SerialRotoscope.cpp Utils.cpp
HEADERS = videoIO.h

#%.o: %.c $(HEADERS)
#	$(CC) $(CFLAGS) -c $< -o $@

#.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(PFLAGS) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *_roto.avi
	-rm -f $(TARGET)
