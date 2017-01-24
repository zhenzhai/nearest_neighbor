CC := g++
SRCDIR := src
TARGET := main
 
SRCEXT := cpp
SOURCES := *.cpp
OBJECTS := $(patsubst .cpp, .o, $(SOURCES))

CFLAGS := -std=c++11 -pthread

ifdef DEBUG
CFLAGS += -g -D DEBUG=$(DEBUG)
endif

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

$(OBJECTS):
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)

new: clean $(TARGET)

directories:
	mkdir output

.PHONY: 
	clean