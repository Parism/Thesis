OMPFLAG = -fopenmp
#PROFLAG = -pg
CC = mpic++
#CC = /home/elias/gcc-bin/bin/g++-4.7.1
NMAX ?= 1002
DEFS = -DNMAX=$(NMAX) -D_PRECALC_
OPTFLAG = -O2 -fomit-frame-pointer -ftree-vectorize -ftree-vectorizer-verbose=1 -msse2 -msse -mfpmath=sse -funroll-loops -fassociative-math -fno-signed-zeros -fno-trapping-math -fno-signaling-nans --param vect-max-version-for-alias-checks=20
FLAGS_SEQ = ${OPTFLAG} ${PROFLAG} ${DEFS} -Wall -g -I../common
FLAGS = ${FLAGS_SEQ} ${OMPFLAG}
LFLAGS_SEQ = ${PROFLAG}
LFLAGS = ${PROFLAG} ${OMPFLAG}
LIBS = -lm -lrt
CLEAN_FILES = lmsor_rb_seq_mpi_only.o lmsor_rb_seq_sse2_mpi_only.o lmsor_rb_omp_mpi_only.o lmsor_rb_omp_sse2_mpi_only.o lmsor_seq_mpi_only lmsor_seq_sse2_mpi_only lmsor_omp_mpi_only lmsor_omp_sse2_mpi_only

all: lmsor_seq_mpi_only lmsor_seq_sse2_mpi_only lmsor_omp_mpi_only lmsor_omp_sse2_mpi_only

lmsor_seq_mpi_only: lmsor_rb_seq_mpi_only.o
	${CC} ${LFLAGS_SEQ} -o $@ lmsor_rb_seq_mpi_only.o ${LIBS}

lmsor_seq_sse2_mpi_only: lmsor_rb_seq_sse2_mpi_only.o
	${CC} ${LFLAGS_SEQ} -o $@ lmsor_rb_seq_sse2_mpi_only.o ${LIBS}

lmsor_omp_mpi_only: lmsor_rb_omp_mpi_only.o
	${CC} ${LFLAGS} -o $@ lmsor_rb_omp_mpi_only.o ${LIBS}

lmsor_omp_sse2_mpi_only: lmsor_rb_omp_sse2_mpi_only.o
	${CC} ${LFLAGS} -o $@ lmsor_rb_omp_sse2_mpi_only.o ${LIBS}

lmsor_rb_seq_mpi_only.o: Lmsor_rb_reord_sse2_mpi_only.cpp
	${CC} -c ${FLAGS_SEQ} -DMODIFIED_SOR Lmsor_rb_reord_sse2_mpi_only.cpp -o $@

lmsor_rb_seq_sse2_mpi_only.o: Lmsor_rb_reord_sse2_mpi_only.cpp
	${CC} -c ${FLAGS_SEQ} -DMODIFIED_SOR -D_INTRINSIC_SSE2_ Lmsor_rb_reord_sse2_mpi_only.cpp -o $@

lmsor_rb_omp_mpi_only.o: Lmsor_rb_reord_sse2_mpi_only.cpp
	${CC} -c ${FLAGS} -DMODIFIED_SOR -DBLOCK_PARTITIONING Lmsor_rb_reord_sse2_mpi_only.cpp -o $@

lmsor_rb_omp_sse2_mpi_only.o: Lmsor_rb_reord_sse2_mpi_only.cpp
	${CC} -c ${FLAGS} -DMODIFIED_SOR -DBLOCK_PARTITIONING -D_INTRINSIC_SSE2_ Lmsor_rb_reord_sse2_mpi_only.cpp -o $@

clean:
	\rm -f $(CLEAN_FILES)

rebuild: clean all
