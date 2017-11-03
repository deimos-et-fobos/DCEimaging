.SUFFIXES: .cpp .c .o

BIN_DIR=../bin/
PROGRAM=DCEbiomarkers
PROGRAM_ORIG=DCEbiomarkers_orig

PROG=$(BIN_DIR)$(PROGRAM)
PROG_ORIG=$(BIN_DIR)$(PROGRAM_ORIG)

OBJS=DCEbiomarkers.o
OBJS_ORIG=DCEbiomarkers_orig.o
FFTDIR=FFT_CODE/
FFTOBJS=$(FFTDIR)complex.o $(FFTDIR)fft.o
LIBS=
LIBDIR=
CFLAGS= -Wall -g -c 

.cpp.o:
	g++ $(CFLAGS) $<

.c.o:
	g++ $(CFLAGS) $<

all: $(PROGRAM) #$(PROGRAM_ORIG)

$(PROGRAM): $(OBJS) $(FFTOBJS) FFTsrc
	g++ -g -o $(PROG) $(OBJS) $(FFTOBJS) $(LIBDIR)$(LIBS)

$(PROGRAM_ORIG): $(OBJS_ORIG) $(FFTOBJS) FFTsrc
	g++ -g -o $(PROG_ORIG) $(OBJS_ORIG) $(FFTOBJS) $(LIBDIR)$(LIBS)

FFTsrc:
	(cd $(FFTDIR); make)

clean: 
		rm -f $(PROG) *~ *.o *.out

wipe: clean
		(cd $(FFTDIR); make clean)
