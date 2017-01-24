CC := g++
SRCDIR := src
TARGET := main
 
SRCEXT := cpp
SOURCES := main.cpp
OBJECTS := main.o

CFLAGS := -std=c++11 -pthread

ifdef DEBUG
CFLAGS += -g -DDEBUG=$(DEBUG)
endif

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(TARGET)

$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) -r $(BUILDDIR) $(TARGET)

new: clean $(TARGET)

directories:
	mkdir output

.PHONY: 
	clean