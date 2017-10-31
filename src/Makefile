.SUFFIXES: .cpp .c .o

BIN_DIR=../bin/
PROGRAM=DCEbiomarkers

PROG=$(BIN_DIR)$(PROGRAM)

OBJS=DCEbiomarkers.o
FFTDIR=FFT_CODE/
FFTOBJS=$(FFTDIR)complex.o $(FFTDIR)fft.o
LIBS=
LIBDIR=
CFLAGS= -Wall -g -c 

.cpp.o:
	g++ $(CFLAGS) $<

.c.o:
	g++ $(CFLAGS) $<

all: $(PROGRAM)

$(PROGRAM): $(OBJS) $(FFTOBJS) FFTsrc
	g++ -g -o $(PROG) $(OBJS) $(FFTOBJS) $(LIBDIR)$(LIBS)

FFTsrc:
	(cd $(FFTDIR); make)

clean: 
		rm -f $(PROG) *~ *.o *.out

wipe: clean
		(cd $(FFTDIR); make clean)
