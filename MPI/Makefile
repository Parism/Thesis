
OMPFLAG = -fopenmp
#PROFLAG = -pg
CC = mpicxx
#CC = /home/elias/gcc-bin/bin/g++-4.7.1
NMAX ?= 8400
DEFS = -DNMAX=$(NMAX) -D_PRECALC_
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=1 -msse2 -msse -mfpmath=sse -funroll-loops -fassociative-math -fno-signed-zeros -fno-trapping-math -fno-signaling-nans --param vect-max-version-for-alias-checks=20
FLAGS_SEQ = ${OPTFLAG} ${PROFLAG} ${DEFS} -Wall -g -I../common
FLAGS = ${FLAGS_SEQ} ${OMPFLAG}
LFLAGS_SEQ = ${PROFLAG}
LFLAGS = ${PROFLAG} ${OMPFLAG}
LIBS = -lrt
MPIP = -L/apps/parallel/mpiP/3.4.1/lib -lmpiP -lm -lbfd -liberty -lunwind
CLEAN_FILES = lmsor_rb_seq_hybrid.o lmsor_rb_seq_sse2_hybrid.o lmsor_rb_omp_hybrid.o lmsor_rb_omp_sse2_hybrid.o lmsor_seq_hybrid lmsor_seq_sse2_hybrid lmsor_omp_hybrid lmsor_omp_sse2_hybrid

all: lmsor_seq_hybrid lmsor_seq_sse2_hybrid lmsor_omp_hybrid lmsor_omp_sse2_hybrid

lmsor_seq_hybrid: lmsor_rb_seq_hybrid.o
	${CC} ${LFLAGS_SEQ} -g -o $@ lmsor_rb_seq_hybrid.o ${LIBS} ${MPIP}

lmsor_seq_sse2_hybrid: lmsor_rb_seq_sse2_hybrid.o
	${CC} ${LFLAGS_SEQ} -o $@ lmsor_rb_seq_sse2_hybrid.o ${LIBS} ${MPIP}

lmsor_omp_hybrid: lmsor_rb_omp_hybrid.o
	${CC} ${LFLAGS} -o $@ lmsor_rb_omp_hybrid.o ${LIBS} ${MPIP}

lmsor_omp_sse2_hybrid: lmsor_rb_omp_sse2_hybrid.o
	${CC} ${LFLAGS} -o $@ lmsor_rb_omp_sse2_hybrid.o ${LIBS} ${MPIP}

lmsor_rb_seq_hybrid.o: Lmsor_rb_reord_mpi.cpp
	${CC} -c ${FLAGS_SEQ} -DMODIFIED_SOR Lmsor_rb_reord_mpi.cpp -o $@

lmsor_rb_seq_sse2_hybrid.o: Lmsor_rb_reord_mpi.cpp
	${CC} -c ${FLAGS_SEQ} -DMODIFIED_SOR -D_INTRINSIC_SSE2_ Lmsor_rb_reord_mpi.cpp -o $@

lmsor_rb_omp_hybrid.o: Lmsor_rb_reord_mpi.cpp
	${CC} -c ${FLAGS} -DMODIFIED_SOR -DBLOCK_PARTITIONING Lmsor_rb_reord_mpi.cpp -o $@

lmsor_rb_omp_sse2_hybrid.o: Lmsor_rb_reord_mpi.cpp
	${CC} -c ${FLAGS} -DMODIFIED_SOR -DBLOCK_PARTITIONING -D_INTRINSIC_SSE2_ Lmsor_rb_reord_mpi.cpp -o $@

clean:
	\rm -f $(CLEAN_FILES)

rebuild: clean all
