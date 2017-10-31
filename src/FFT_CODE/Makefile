.SUFFIXES: .cpp .c .o

LIBS=
LIBDIR=
OBJS=fft.o complex.o
CFLAGS= -Wall -g -c 

.cpp.o: 
	g++ $(CFLAGS) $<

.c.o: 
	g++ $(CFLAGS) $<

all: $(OBJS) 

clean: 
		rm -f *~ *.o *.out

wipe: clean
