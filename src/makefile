CC := g++
SRCDIR := src
BUILDDIR := build
TARGET := main.o
 
SRCEXT := cpp
SOURCE := main.cpp
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -std=c++11 -pthread

$(TARGET):
	$(CC) $(CFLAGS) $(SRCDIR)/$(SOURCE) -o $(TARGET)