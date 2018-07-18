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
CFLAGS = -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = main.cpp
HEADERS =

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *_roto.avi
	-rm -f $(TARGET)
