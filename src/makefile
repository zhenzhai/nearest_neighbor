CC := g++ 
SRCDIR := src
BUILDDIR := build
TARGET := main.o
 
SRCEXT := cpp
SOURCE := main.cpp
OBJECTS := $(patsubst $(SRCDIR)/%, $(BUILDDIR)/%, $(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -std=c++11 -pthread 

ifdef DEBUG
CFLAGS += -g -DDEBUG=$(DEBUG)
endif

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(SRCDIR)/$(SOURCE) -o $(TARGET)
