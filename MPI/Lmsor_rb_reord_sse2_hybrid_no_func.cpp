/* Local Modified SOR_5pt (LMSOR_RB)  Missirlis and Tzaferis (3.11.2002)*/
/* red\black ordering */
/* optimum  w1_ij, w2_ij */
/* Solving of the second order Convection Diffusion PDE */
/*---------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include "timestamp.h"
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>

#else
inline int omp_get_num_threads(void){return 1;}
inline int omp_get_thread_num(void){return 0;}
#endif

#if (defined __SSE2__) || (_M_IX86_FP == 2)
#include <emmintrin.h>
#define __SIMD_SUPPORTED__
#endif

#define ERROR_CHECK_ITERS 100
#define CHECK_SQRERROR 1

#ifndef NMAX
#define NMAX 1000
#endif


// -------------------
// Allowed definitions:
// #define NMAX 1002
// #define MODIFIED_SOR
// #define _PRECALC_
// #define _INTRINSIC_SSE2_
// #define BLOCK_PARTITIONING
// -------------------

#if defined(_INTRINSIC_SSE2_) && (!defined(_PRECALC_) || !defined(__SIMD_SUPPORTED__))
#error "Can not use _INTRINSIC_SSE2_ without _PRECALC_ and __SIMD_SUPPORTED__"
#endif

#ifdef _MSC_VER
#define isnan(x) _isnan(x)  // VC++ uses _isnan() instead of isnan()
#endif

#define _DBLS_PER_CACHE_LINE_ (64/sizeof(double)) // count of doubles that fit in a cache line
//#define _ALIGN_SZ_(x,y) ((x+y-1)/y)*y
#define _ALIGN_SZ_(x,y) x/2

#ifdef _OPENMP
#ifndef BLOCK_PARTITIONING
#error "Only block partitioning is implemented"
#endif
#endif



inline void mypause(void){
  printf("Press \"Enter\"\n");
  timestamp ts = getTimestamp();
  do {
    getchar();
  } while( getElapsedtime(ts)<100.0f );
}

inline long int pow(int b, int e){
	long int r = 1;
	for(int i=0; i<e; i++)
		r *= b;
	return r;
}

inline double pow2(double v){
	return v*v;
}

inline void validate_w(double w){
	if( isnan(w) ){
		printf("ERROR: Invalid w value\n");
		exit(1);
	}
}


class regular_array{
	double *av; //MAX MAX
	int size;
public:
	
	/*~regular_array(){
		delete[] av;
	}*/
	inline void set(int row, int col, double v){
		*(av+(row*(this->size))+col) = v;
	}
	inline double get(int row, int col){
		return *(av+(row*(this->size))+col);
	}
	inline double* _get_data(void){
		return &av[0];
	}
	inline double** _get_av(void){
		return &av;
	}
	inline void set_size(int size){
		this->size = size;
	}
	inline int get_size(){
		return this->size;
	}
	/*
	inline void allocate_array(){
		fprintf(stderr, "size = %d\n", this->size );
		av = new double*[this->size];
		for(int i = 0; i<this->size; i++){
			av[i] = new double[this->size];
		}
	}*/
};


class redblack_array{
	double *av;  //[2][MAX][_ALIGN_SZ_(MAX/2, _DBLS_PER_CACHE_LINE_)]
	int size;
public:
	
	/*~redblack_array(){
		delete[] av;
	}*/
	inline void set(int row, int col, double v){
		if( (row+col)%2==0 )
			*(av+(row*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2))+(col/2)) = v;
		else
			*(av+((this->size)*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2))+(row*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2))+(col/2)) = v;
	}
	inline double get(int row, int col){
		if( (row+col)%2==0 )
			return *(av+(row*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2))+col/2);
		else
			return *(av+((this->size)*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2))+(row*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2))+(col/2));
	}
	/*inline double**** _get_array(){
		return &av;
	}*/
	inline double* _get_data(int i){
		return av+i*((this->size)*((_ALIGN_SZ_((this->size), _DBLS_PER_CACHE_LINE_))+2));
	}

	inline double* get_pointer(){
		return &av[0];
	}


	inline double** get_av(){
		return &av;
	}
	inline void set_size(int size){
		this->size = size;
	}
	inline int get_size(){
		return this->size;
	}
	inline void copyfrom(regular_array &arr){
		for(int iy=0; iy<(get_size()); iy++)
			for(int ix=0; ix<(get_size()); ix++)
				set(iy, ix, arr.get(iy, ix));
	}
	inline void copyto(regular_array &arr){
		for(int iy=0; iy<(get_size()); iy++)
			for(int ix=0; ix<(get_size()); ix++)
				arr.set(iy, ix, get(iy, ix));
	}
/*
	inline void allocate_array(){
		av = new double**[2];
		for(int i = 0; i<2; i++){
			av[i] = new double*[this->size];
			for(int y=0; y<this->size; y++){
				av[i][y] = new double[_ALIGN_SZ_((this->size)/2, _DBLS_PER_CACHE_LINE_)];
			}
		}
	}*/
};


	inline void send_recv_init(double* av,int up,int down,int left,int right,MPI_Comm new_comm,MPI_Datatype col_type,MPI_Request* reqs,int *tags,int block_size){

	

	MPI_Send_init( av+block_size+1, block_size-2, MPI_DOUBLE, up, tags[1], new_comm, reqs); //send first row of initial array
	MPI_Send_init( av+(block_size)*(block_size-2)+1, block_size-2, MPI_DOUBLE, down, tags[0], new_comm, reqs+1 );  //send last row of initial array
	MPI_Send_init( av+block_size+1, 1, col_type, left, tags[3], new_comm, reqs+2 );			//send left column 
	MPI_Send_init( av+block_size+block_size-2, 1, col_type, right, tags[2], new_comm, reqs+3 );		//send right column

	MPI_Recv_init( av+1, block_size-2, MPI_DOUBLE, up, tags[0], new_comm, reqs+4 );		
	MPI_Recv_init( av+(block_size)*(block_size-1)+1, block_size-2, MPI_DOUBLE, down, tags[1], new_comm, reqs+5 );
 	MPI_Recv_init( av+block_size, 1,col_type, left, tags[2],new_comm, reqs+6 );
 	MPI_Recv_init( av+block_size+block_size-1, 1, col_type, right, tags[3], new_comm, reqs+7);


/*
 	MPI_Send_init( &(av[1][1]), block_size-2, MPI_DOUBLE, up, tags[1], new_comm, reqs); //send first row of initial array
	MPI_Send_init( &(av[block_size-2][1]), block_size-2, MPI_DOUBLE, down, tags[0], new_comm, reqs+1 );  //send last row of initial array
	MPI_Send_init( &(av[1][1]), 1, col_type, left, tags[3], new_comm, reqs+2 );			//send left column 
	MPI_Send_init( &(av[1][block_size-2]), 1, col_type, right, tags[2], new_comm, reqs+3 );		//send right column

	MPI_Recv_init( &(av[0][1]), block_size-2, MPI_DOUBLE, up, tags[0], new_comm, reqs+4 );		
	MPI_Recv_init( &(av[block_size-1][1]), block_size-2, MPI_DOUBLE, down, tags[1], new_comm, reqs+5 );
 	MPI_Recv_init( &(av[1][0]), 1,col_type, left, tags[2],new_comm, reqs+6 );
 	MPI_Recv_init( &(av[1][block_size-1]), 1, col_type, right, tags[3], new_comm, reqs+7);
*/
 	//fprintf(stderr, "req prwto mesa sti synartisi = %d\n",*(reqs) );

 	return;
}

	inline void send_recv_init_rb(double* av,int up,int down,int left,int right,MPI_Comm new_comm,MPI_Datatype col_type,MPI_Request* reqs,int *tags,int block_size){

	int columns = _ALIGN_SZ_(block_size, _DBLS_PER_CACHE_LINE_);

	columns+=2;

	MPI_Send_init( av+columns+1, columns-2, MPI_DOUBLE, up, tags[1], new_comm, reqs); //send first row of initial array
	MPI_Send_init( av+(columns)*(block_size-2)+1, columns-2, MPI_DOUBLE, down, tags[0], new_comm, reqs+1 );  //send last row of initial array
	MPI_Send_init( av+block_size+1, 1, col_type, left, tags[3], new_comm, reqs+2 );			//send left column 
	MPI_Send_init( av+columns+columns-2, 1, col_type, right, tags[2], new_comm, reqs+3 );		//send right column

	MPI_Recv_init( av+1, block_size-2, MPI_DOUBLE, up, tags[0], new_comm, reqs+4 );		//receive first row
	MPI_Recv_init( av+(columns)*(block_size-1)+1, block_size-2, MPI_DOUBLE, down, tags[1], new_comm, reqs+5 ); //receive last row
 	MPI_Recv_init( av+columns, 1,col_type, left, tags[2],new_comm, reqs+6 ); //receive left column
 	MPI_Recv_init( av+columns+columns-1, 1, col_type, right, tags[3], new_comm, reqs+7); //receive right column


/*
 	MPI_Send_init( &(av[1][1]), block_size-2, MPI_DOUBLE, up, tags[1], new_comm, reqs); //send first row of initial array
	MPI_Send_init( &(av[block_size-2][1]), block_size-2, MPI_DOUBLE, down, tags[0], new_comm, reqs+1 );  //send last row of initial array
	MPI_Send_init( &(av[1][1]), 1, col_type, left, tags[3], new_comm, reqs+2 );			//send left column 
	MPI_Send_init( &(av[1][block_size-2]), 1, col_type, right, tags[2], new_comm, reqs+3 );		//send right column

	MPI_Recv_init( &(av[0][1]), block_size-2, MPI_DOUBLE, up, tags[0], new_comm, reqs+4 );		
	MPI_Recv_init( &(av[block_size-1][1]), block_size-2, MPI_DOUBLE, down, tags[1], new_comm, reqs+5 );
 	MPI_Recv_init( &(av[1][0]), 1,col_type, left, tags[2],new_comm, reqs+6 );
 	MPI_Recv_init( &(av[1][block_size-1]), 1, col_type, right, tags[3], new_comm, reqs+7);
*/
 	//fprintf(stderr, "req prwto mesa sti synartisi = %d\n",*(reqs) );

 	return;
}

void my_start_all(MPI_Request* reqs){

		MPI_Start(reqs);
		MPI_Start(reqs+1);
		MPI_Start(reqs+2);
		MPI_Start(reqs+3);
		MPI_Start(reqs+4);
		MPI_Start(reqs+5);
		MPI_Start(reqs+6);
		MPI_Start(reqs+7);


	return;
}

void my_wait_all(MPI_Request *reqs){

		MPI_Wait(reqs,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+1,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+2,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+3,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+4,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+5,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+6,MPI_STATUSES_IGNORE);
		MPI_Wait(reqs+7,MPI_STATUSES_IGNORE);


		return;
}

regular_array arr_u;
#ifdef _PRECALC_
regular_array arr_ffh, arr_ggh;
#else
regular_array arr_l, arr_r, arr_t, arr_b;
#endif
regular_array arr_w;

redblack_array ro_u;
#ifdef _PRECALC_
redblack_array ro_ffh, ro_ggh;
#else
redblack_array ro_l, ro_r, ro_t, ro_b;
#endif
redblack_array ro_w;

double FF(int epil, double r1, double x1, double y1);
double GG(int epil, double r1, double x1, double y1);
double initial_guess(double x1, double y1);
void min_max_MAT(double MAT[], int n, double *min_MAT, double *max_MAT , int block_size);

inline double calcPoint(const int pitch, const int line_indicator, double *pu_point, const double *pu_neighb, const double *pffh, const double *pggh, const double *pw){
	double r=(1.-*pffh)/4.;
	double l=(1.+*pffh)/4.;
	double t=(1.-*pggh)/4.;
	double b=(1.+*pggh)/4.;
	*pu_point = ( 1. - *pw )* *pu_point + *pw *( l * *(pu_neighb-pitch) + r * *(pu_neighb+pitch) + b * *(pu_neighb-line_indicator) + t * *(pu_neighb+1-line_indicator) );
	return *pu_point;
}

inline double calcPointUpdErr(const int pitch, const int line_indicator, double *pu_point, const double *pu_neighb, const double *pffh, const double *pggh, const double *pw, double *sqrerror){
	double old_val = *pu_point;
	double new_val = calcPoint(pitch, line_indicator, pu_point, pu_neighb, pffh, pggh, pw);
	double sqr_diff = pow2(old_val - new_val);
	*sqrerror += sqr_diff;
	return new_val;
}

template<int phase>
inline void calcSegment(const int bound_s_x, const int bound_e_x, const int bound_s_y, const int bound_e_y, double &sqrerror, int block_size, int taskid, int it){

#ifdef _INTRINSIC_SSE2_

	const int pitch = _ALIGN_SZ_((block_size-2), _DBLS_PER_CACHE_LINE_);
	const int SIMD_WIDTH = sizeof(__m128d) / sizeof(double);
	double * __restrict pu_des = ro_u._get_data(phase);
	const double * __restrict pu_oth = ro_u._get_data(1-phase);
	const double * __restrict pw = ro_w._get_data(phase);
	const double * __restrict pffh = ro_ffh._get_data(phase);
	const double * __restrict pggh = ro_ggh._get_data(phase);
	const __m128d m_one = _mm_set1_pd(1.0), m_quarter = _mm_set1_pd(0.25);
	for(int i=bound_s_y; i<bound_e_y; i++){
		int line_indicator = (1-phase+i)%2;

		int color_offset  = (i+bound_s_x+phase) % 2;
		int line_idx_from = i*pitch + (bound_s_x + color_offset)/2;
		int line_idx_to   = i*pitch + bound_e_x/2 + ((bound_s_x + color_offset)%2<(bound_e_x%2));

//printf("%d. start %p end %p\n", phase, u._get_data(phase)+line_idx_from, u._get_data(phase)+line_idx_to);
		if( ((size_t)(ro_u._get_data(phase)+line_idx_from)) % sizeof(__m128d) ){
			calcPointUpdErr(pitch, line_indicator, &pu_des[line_idx_from], &pu_oth[line_idx_from], &pffh[line_idx_from], &pggh[line_idx_from], &pw[line_idx_from], &sqrerror);
			line_idx_from++;
		}
		int do_last = ((size_t)(ro_u._get_data(phase)+line_idx_to)) % sizeof(__m128d);
		if( do_last )
			line_idx_to--;


		for(int idx=line_idx_from; idx<line_idx_to; idx+=SIMD_WIDTH){
			__m128d m_ffh = _mm_load_pd(&pffh[idx]);
			__m128d m_ggh = _mm_load_pd(&pggh[idx]);
			__m128d m_opr, m_op2, m_u_n, m_sum;
			// left
			m_u_n = _mm_load_pd(&pu_oth[idx-pitch]);
			m_opr = _mm_add_pd(m_one, m_ffh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_sum = _mm_mul_pd(m_u_n, m_opr);
			// right
			m_u_n = _mm_load_pd(&pu_oth[idx+pitch]);
			m_opr = _mm_sub_pd(m_one, m_ffh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_u_n = _mm_mul_pd(m_u_n, m_opr);
			m_sum = _mm_add_pd(m_u_n, m_sum);
			// bottom
			m_u_n = _mm_loadu_pd(&pu_oth[idx-line_indicator]);
			m_opr = _mm_add_pd(m_one, m_ggh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_u_n = _mm_mul_pd(m_u_n, m_opr);
			m_sum = _mm_add_pd(m_u_n, m_sum);
			// top
			m_u_n = _mm_loadu_pd(&pu_oth[idx+1-line_indicator]);
			m_opr = _mm_sub_pd(m_one, m_ggh);
			m_opr = _mm_mul_pd(m_quarter, m_opr);
			m_u_n = _mm_mul_pd(m_u_n, m_opr);
			m_sum = _mm_add_pd(m_u_n, m_sum);
			// multiply with w_ij
			m_opr = _mm_load_pd(&pw[idx]);
			m_sum = _mm_mul_pd(m_opr, m_sum);
			// first term
			m_op2 = _mm_sub_pd(m_one, m_opr);
			m_u_n = _mm_load_pd(&pu_des[idx]);
			m_opr = _mm_mul_pd(m_op2, m_u_n);
			m_sum = _mm_add_pd(m_opr, m_sum);

			_mm_store_pd(&pu_des[idx], m_sum);
//			_mm_stream_pd(&pu_des[idx], m_sum);
//			_mm_store_sd(&v, m_sum);
/*			_mm_storel_pd(&v, m_sum);
			double max1 = fabs(v);
			if( tmax<max1 ) tmax = max1;
			_mm_storeh_pd(&v, m_sum);
			max1 = fabs(v);
			if( tmax<max1 ) tmax = max1;*/

			m_opr = _mm_sub_pd(m_u_n, m_sum);
			m_sum = _mm_mul_pd(m_opr, m_opr);

			__m128d m_perm = _mm_shuffle_pd (m_sum, m_sum, 1);
			m_perm = _mm_add_pd(m_sum, m_perm);
			double sqr_diff;
			_mm_storel_pd(&sqr_diff, m_perm);
//			double sqr_diff = fabs(v);
//			if( sqrerror<max1 ) sqrerror = max1;
			sqrerror += sqr_diff;
		}

		if( do_last ){
			calcPointUpdErr(pitch, line_indicator, &pu_des[line_idx_to], &pu_oth[line_idx_to], &pffh[line_idx_to], &pggh[line_idx_to], &pw[line_idx_to], &sqrerror);
		}
	}
//	_mm_sfence();

#else
	for(int i=bound_s_y; i<bound_e_y; i++){
		for(int j=bound_s_x; j<bound_e_x; j++){
			if ( (i+j)%2 == phase ) {
				double old_val = ro_u.get(i, j);
#ifdef _PRECALC_
				double r=1.-ro_ffh.get(i, j);
				double l=1.+ro_ffh.get(i, j);
				double t=1.-ro_ggh.get(i, j);
				double b=1.+ro_ggh.get(i, j);

				const double D=1./4.;
				r=r*D;
				l=l*D;
				t=t*D;
				b=b*D;
				
				ro_u.set(i, j, ( 1. - ro_w.get(i, j) )*ro_u.get(i, j) + ro_w.get(i, j)*( l*ro_u.get(i-1, j) + r*ro_u.get(i+1, j) + b*ro_u.get(i, j-1) + t*ro_u.get(i, j+1) ) );
#else
				
				ro_u.set(i, j, ( 1. - ro_w.get(i, j) )*ro_u.get(i, j) + ro_w.get(i, j)*( ro_l.get(i, j)*ro_u.get(i-1, j) + ro_r.get(i, j)*ro_u.get(i+1, j) + ro_b.get(i, j)*ro_u.get(i, j-1) + ro_t.get(i, j)*ro_u.get(i, j+1) ) );
#endif
				sqrerror += pow2(old_val - ro_u.get(i, j));
				
				
//				double max1 = fabs(ro_u.get(i, j));
//				if( tmax<max1 ) tmax = max1;
			}
			
		}
	}
#endif
//fprintf(stderr,"TELOS TIS CALCSEGMENTTT!!!! eimai to task me id= %d\n",taskid);
}

// Zeroes all values in involved matrices by using the same thread assignment used during computation
// so memory is local to assigned processors on NUMA architectures
void firstTouch(int block_size,int num_of_threads){
	
#pragma omp parallel num_threads(num_of_threads)
	{
#pragma omp master
		printf("touching data for first time\n");
		int num_threads = omp_get_num_threads();
		int sy=num_threads, sx=1, sx_limit=(int)sqrt((double)sy);
		for(int i=sx_limit; i>0; i--)
		if( num_threads%i==0 ){
			sx = i;
			sy = num_threads / i;
			break;
		}
		int n=((block_size)-1);
		int bs_x=n/sx, bs_y=n/sy, lastoffset_x=n%sx, lastoffset_y=n%sy;
		int start_pos_x = 1+bs_x * (omp_get_thread_num()%sx), start_pos_y=1+bs_y * (omp_get_thread_num()/sx), end_pos_x=start_pos_x+bs_x+(omp_get_thread_num()%sx==sx-1?lastoffset_x:0), end_pos_y=start_pos_y+bs_y+(omp_get_thread_num()/sx==sy-1?lastoffset_y:0);

		//for(int i=start_pos_y; i<end_pos_y; i++)
		for(int i=start_pos_y; i<end_pos_y; i++)
			for(int j=start_pos_x; j<end_pos_x; j++){
				arr_u.set(i, j, 0.0);
				ro_u.set(i, j, 0.0);
//				u.set_buf(i, j, 0.0);
				arr_w.set(i, j, 0.0);
				ro_w.set(i, j, 0.0);
#ifdef _PRECALC_
				arr_ffh.set(i, j, 0);
				ro_ffh.set(i, j, 0);
				arr_ggh.set(i, j, 0);
				ro_ggh.set(i, j, 0);
#else
				arr_l.set(i, j, 0);
				arr_r.set(i, j, 0);
				arr_t.set(i, j, 0);
				arr_b.set(i, j, 0);
				ro_l.set(i, j, 0);
				ro_r.set(i, j, 0);
				ro_t.set(i, j, 0);
				ro_b.set(i, j, 0);
#endif
//				d.set(i, j, 0);
			}
	}
/*#pragma omp parallel for
	for(int i=0; i<NMAX; i++)
      for(int j=0; j<NMAX; j++) {
		u.set(i, j, 0.0);
//		u.set_buf(i, j, 0.0);
		w.set(i, j, 0.0);
#ifdef _PRECALC_
				ffh.set(i, j, 0);
				ggh.set(i, j, 0);
#else
				l.set(i, j, 0);
				r.set(i, j, 0);
				t.set(i, j, 0);
				b.set(i, j, 0);
#endif
      }*/

}

int build_omega(FILE *arxeio, double re, long &cI, long &cR, double &min_w1, double &max_w1, double &min_w2, double &max_w2, int epilogi, double &min_m_low, double &max_m_low, double &min_m_up, double &max_m_up, int block_size,int taskid,int num_of_threads){
	double *w1 = new double[block_size*block_size];
	double *w2 = new double[block_size*block_size];
	double *m_low = new double[block_size*block_size];
	double *m_up = new double[block_size*block_size];
	
	int nn=block_size-1, i, j, periptosi=0, tperiptosi=0;
	int gperiptosi=0;
	double x, y, C_E,C_W,C_N,C_S,g1,g2;
	long cImags = 0, cReals = 0;
	double pi=4.0*atan(1.);

	int n=nn-1;
    double h=1./(n+1.0);
//    printf("Starting up...");
    timestamp ts_start = getTimestamp();
#pragma omp parallel for private(j,x,y,C_E,C_W,C_N,C_S,g1,g2) firstprivate(periptosi, tperiptosi) reduction(+ : cImags, cReals) num_threads(num_of_threads)
    for(i=0; i<=n+1; i++) {
      x=i*h;
      for(j=0; j<=n+1; j++) {
        y=j*h;

#ifdef _PRECALC_ 
		arr_ffh.set(i, j, (1./2.)*h*FF(epilogi,re,x,y));
		
		arr_ggh.set(i, j, (1./2.)*h*GG(epilogi,re,x,y));
		C_E=(1-arr_ffh.get(i, j))/4.;
		C_W=(1+arr_ffh.get(i, j))/4.;
		C_N=(1-arr_ggh.get(i, j))/4.;
		C_S=(1+arr_ggh.get(i, j))/4.;
#else
		arr_r.set(i, j, 1-(1./2.)*h*FF(epilogi,re,x,y));     /* u[i+1][j]  */
		arr_l.set(i, j, 1+(1./2.)*h*FF(epilogi,re,x,y));     /* u[i-1][j]  */
		arr_t.set(i, j, 1-(1./2.)*h*GG(epilogi,re,x,y));     /* u[i][j+1]  */
		arr_b.set(i, j, 1+(1./2.)*h*GG(epilogi,re,x,y));     /* u[i][j-1]  */
		arr_r.set(i, j, arr_r.get(i, j)/4.);
		arr_l.set(i, j, arr_l.get(i, j)/4.);
		arr_t.set(i, j, arr_t.get(i, j)/4.);
		arr_b.set(i, j, arr_b.get(i, j)/4.);
		C_E=arr_r.get(i, j);
		C_W=arr_l.get(i, j);
		C_N=arr_t.get(i, j);
		C_S=arr_b.get(i, j);
#endif

        /* ektimhsh tou w_opt  */
        if (C_E*C_W*C_N*C_S>=0) {
          if ( C_E*C_W>=0 && C_N*C_S>=0 )
            periptosi=1;
          else
            if ( C_E*C_W<=0 && C_N*C_S<=0 )
              periptosi=2;
        } else {
          if ( C_E*C_W>0 && C_E+C_W>=0)
            periptosi=3;
          else
            if ( C_E*C_W<0 && C_N+C_S>=0 )
              periptosi=4;
            else
              periptosi=5;
        }
		if(tperiptosi==0)
			tperiptosi = periptosi;
		else if(tperiptosi!=periptosi)
			tperiptosi = -1;

        switch(periptosi) {
          case 1:
            /* case 1 (Im(m)=0) real */
            m_up[i*block_size+j] = 2.*(sqrt(C_E*C_W)+sqrt(C_N*C_S))*cos(pi*h);
            m_low[i*block_size+j] = 2.*(sqrt(C_E*C_W)+sqrt(C_N*C_S))*cos(pi*(1.-h)/2.);
//printf("(%f) %f\n",2.*(sqrt(C_E*C_W)+sqrt(C_N*C_S)),m_low[i][j]);exit(1);
#ifdef MODIFIED_SOR
            w1[i*block_size+j] = 2./(1.-m_up[i*block_size+j]*m_low[i*block_size+j]+sqrt((1.-m_up[i*block_size+j])*(1.-m_low[i*block_size+j])));
            w2[i*block_size+j] = 2./(1.+m_up[i*block_size+j]*m_low[i*block_size+j]+sqrt((1.-m_up[i*block_size+j])*(1.-m_low[i*block_size+j])));
#else
			w1[i*block_size+j] = w2[i*block_size+j] = 2./(1.+sqrt(1.-m_up[i*block_size+j]*m_up[i*block_size+j]));
#endif
			cReals++;
            break;

          case 2:
            /* case 2 (Re(m)=0) imaginary */
            m_up[i*block_size+j] = 2.*(sqrt(-C_E*C_W)+sqrt(-C_N*C_S))*cos(pi*h);
            m_low[i*block_size+j] = 2.*(sqrt(-C_E*C_W)+sqrt(-C_N*C_S))*cos(pi*(1.-h)/2.);

#ifdef MODIFIED_SOR
            w1[i*block_size+j] = 2./(1.-m_up[i*block_size+j]*m_low[i*block_size+j]+sqrt((1.+pow(m_up[i*block_size+j],2.0))*(1.+pow(m_low[i*block_size+j],2.0))));
            w2[i*block_size+j] = 2./(1.+m_up[i*block_size+j]*m_low[i*block_size+j]+sqrt((1.+pow(m_up[i*block_size+j],2.0))*(1.+pow(m_low[i*block_size+j],2.0))));
#else
			w1[i*block_size+j] = w2[i*block_size+j] = 2./(1.+sqrt(1.+m_up[i*block_size+j]*m_up[i*block_size+j]));
#endif
			cImags++;
            break;

          case 3:
            /* case 3a  complex  */
            g1=pow((1.-pow((C_E+C_W), 2./3.)),-1./2.);
            w1[i*block_size+j]=2./(1.+g1*fabs(C_N-C_S));
            break;

          case 4:
            /* case 3b complex  */
            g2=pow((1.-pow((C_N+C_S),2./3.)),-1./2.);
            w1[i*block_size+j]=2./(1.+g2*fabs(C_E-C_W));
            w2[i*block_size+j]=w1[i*block_size+j];
            break;

          case 5:
            /* alli periptosi */
            printf("\n####h methodos diakoptetai####\n");
            fprintf(arxeio,"\n###### h methodos diakoptetai #######\n");
            break;

          default:
            printf("\n lathos periptosi");
            break;
        }
		validate_w(w1[i*block_size+j]);
		validate_w(w2[i*block_size+j]);
      } /* end for j */
		
#pragma omp critical
      {
		  if( tperiptosi!=0 && gperiptosi!=-1 && tperiptosi!=gperiptosi ){
			  if( gperiptosi!=0 )
			    gperiptosi = -1;
			  else
			    gperiptosi = tperiptosi;
		  }
	  }
    } /*end for i */
    printf(" %f seconds\n", getElapsedtime(ts_start)/1000.0);

    switch(gperiptosi) {
		case 1:
			printf("\n ---Pragmatikh periptosi---- \n");
			break;
		case 2:
			printf("\n ---Fantastikh periptosi---- \n");
			break;
		default:
			printf("\n ---Mikth periptosi---- \n");
			break;
    }


	min_max_MAT(w1, n,  &min_w1, &max_w1,block_size);
    min_max_MAT(w2, n,  &min_w2, &max_w2,block_size);
    min_max_MAT(m_low, n,  &min_m_low, &max_m_low,block_size);
    min_max_MAT(m_up, n, &min_m_up, &max_m_up, block_size);
    printf("PDE :  %2d      h =  %lf     n = %6d \n",epilogi,h,n);
    printf("\n periptosi:  %2d RANGE_m_low = %5.4lf - %5.4lf \t RANGE_m_up = %5.4lf - %5.4lf\n",
           periptosi,min_m_low,max_m_low,min_m_up,max_m_up);

    fprintf(arxeio, "\n  %3d  \t  %5.4lf - %5.4lf \t %5.4lf - %5.4lf\n",
                    periptosi,min_m_low,max_m_low,min_m_up,max_m_up);

	// create unified matrix w
    for(i=0; i<=n+1; i++) {
      for(j=0; j<=n+1; j++) {
		arr_w.set(i,j, ( (i+j)%2 == 0 ) ? w1[(i*block_size)+j] : w2[(i*block_size)+j]);
		
	  }
	}

	cI = cImags;
	cR = cReals;
	delete []w1;
	delete []w2;
	delete []m_low;
	delete []m_up;
	return gperiptosi;
}

int main(int argc, char* argv[]) {

  long i,j,k,metr;
  k=1;
  int epilogi, periptosi=0, ektyp1, ektyp2;
  long maxiter;
  	//int nn=NMAX-1;
	//int n=nn-1;
//  double D;
  double min_w1, max_w1, min_w2, max_w2, min_m_low, max_m_low, min_m_up, max_m_up;
  double x, y;
   // double h=1./(n+1.0);
//  double C_E, C_W, C_N, C_S, g1, g2;
  double e1=1.0e-6, sqrerror, re;
  char filename[20];
	timestamp ts_start;
  FILE *arxeio;

  int numtasks,taskid;

#ifdef _PRECALC_
	fprintf(stderr, "EINAI ORISMENOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" );
#endif



#ifdef __SIMD_SUPPORTED__
	printf("SIMD (SSE2) instructions are supported! :)\n");
#ifdef _INTRINSIC_SSE2_
	printf("Using SSE2 hand written intrinsic code...\n");
#endif
#endif

  if( argc != 7 ){
    printf("\nSyntax:\nlmsor_cpu FILENAME MAX_ITERATIONS SELECTION RE PRINT_RANGE PRINT_SOL\n");
  }

  if( argc >= 2 )
    strcpy(filename, argv[1]);
  else{
    printf("\n dose to onoma tou arxeiou ektyposis apotelesmatwn :  ");
    if( scanf("%s",filename) == 0 )
      exit(0);
    /* gets(filename);  */
  }
  printf("Output: %s\n", filename);
  if ((arxeio=fopen(filename,"w"))==NULL) {
    printf("i fopen apetixe \n");
    exit(0);
  }

  if( argc >= 3 )
    maxiter = atoi(argv[2]);
  else{
    printf("megisto epitrepto plithos epanalipsewn ( maxiter ) : ");
    if( scanf("%ld",&maxiter)==0 )
      exit(0);
  }
  printf("\n maxiter = %ld ",maxiter);
  if( argc >= 4 )
    epilogi = atoi(argv[3]);
  else{
    printf("\n epilogi twn syntelestwn ths merikhs diaforikhs exiswsis(PDE) : ");
    if( scanf("%d",&epilogi)==0 )
      exit(0);
  }
  printf("\n epilogi = %d\n",epilogi);
  if( argc >= 5 )
    re = atof(argv[4]);
  else{
    printf("\n re = ");
    if( scanf("%lf", &re)==0 )
      exit(0);
  }
  printf("\n re = %lf\n",re);

  if( argc >= 7 ){
    ektyp1 = atoi(argv[5]);
    ektyp2 = atoi(argv[6]);
  } else {
    printf("\n dose gia ektyposi range_w=1 , ektyposi lisis=1\n");
    if( scanf("%d %d",&ektyp1,&ektyp2)==0 )
      exit(0);
  }

  fprintf(arxeio, "-----------Local MSOR 5-points (LMSOR_RB)------------\n\n\n");
  fprintf(arxeio, " PDE = %2d           Re =  %.lf", epilogi,re);

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  
  int block_size;
  int root = sqrt(numtasks);
  block_size = NMAX/root;
  fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!block_size = %d\n",block_size );

  int nn=block_size-1;
  int n=nn-1;
  double h=1./(n+1.0);
  block_size = block_size+2;
  
  
  arr_u.set_size(block_size);
  arr_w.set_size(block_size);


  double** arr_pointer = arr_u._get_av();
  *arr_pointer = new double[block_size*block_size];

  arr_pointer = arr_w._get_av();
  *arr_pointer = new double[block_size*block_size];



#ifdef _PRECALC_
	arr_ffh.set_size(block_size);
	arr_ggh.set_size(block_size);



	arr_pointer = arr_ffh._get_av();
  	*arr_pointer = new double[block_size*block_size];

  	arr_pointer = arr_ggh._get_av();
  	*arr_pointer = new double[block_size*block_size];
	
#else


	arr_l.set_size(block_size);
	arr_r.set_size(block_size);
	arr_t.set_size(block_size);
	arr_b.set_size(block_size);

	arr_pointer = arr_l._get_av();
  	*arr_pointer = new double[block_size*block_size];

  	arr_pointer = arr_r._get_av();
  	*arr_pointer = new double[block_size*block_size];

  	arr_pointer = arr_t._get_av();
  	*arr_pointer = new double[block_size*block_size];

  	arr_pointer = arr_b._get_av();
  	*arr_pointer = new double[block_size*block_size];

#endif
  
  	int columns = _ALIGN_SZ_(block_size, _DBLS_PER_CACHE_LINE_);

	ro_u.set_size(block_size);
	ro_w.set_size(block_size);

	double** ro_pointer = ro_u.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];

	ro_pointer = ro_w.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];
	

	


#ifdef _PRECALC_
  

	ro_ffh.set_size(block_size);
	ro_ggh.set_size(block_size);

	ro_pointer = ro_ffh.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];

	ro_pointer = ro_ggh.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];
    
    
   

#else
	
	ro_l.set_size(block_size);
	ro_t.set_size(block_size);
	ro_r.set_size(block_size);
	ro_b.set_size(block_size);


	ro_pointer = ro_l.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];

	ro_pointer = ro_r.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];

	ro_pointer = ro_t.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];

	ro_pointer = ro_b.get_av();
	*ro_pointer = new double[2*block_size*(columns+2)];

#endif



#ifdef _OPENMP
	
	firstTouch(block_size,atoi(argv[7]));
#endif



  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  int square_workers = sqrt(numtasks);

  MPI_Datatype col_type;
  MPI_Datatype block_type;
  MPI_Datatype inner_block;
  MPI_Datatype ro_col_type;
  MPI_Type_vector(block_size-2,1,block_size,MPI_DOUBLE,&col_type);
  MPI_Type_vector(block_size-2,block_size-2,NMAX,MPI_DOUBLE,&block_type);
  MPI_Type_vector(block_size-2,block_size-2,block_size,MPI_DOUBLE,&inner_block);
  MPI_Type_vector(block_size-2,1,columns+2,MPI_DOUBLE,&ro_col_type);

  MPI_Type_commit(&col_type);
  MPI_Type_commit(&block_type);
  MPI_Type_commit(&inner_block);
  MPI_Type_commit(&ro_col_type);

  /* Creation of a new communicator */
  int dim_array[2];
  dim_array[0]=square_workers;
  dim_array[1]=square_workers; 
  int periods[2];
  periods[0]=0;
  periods[1]=0;
  MPI_Comm new_comm;  

  MPI_Cart_create( MPI_COMM_WORLD,2,dim_array,periods, 0,&new_comm); 




	// Support omega values input from the environment
	long cImags=0, cReals=0;
	char *envOmega1 = getenv("LMSOR_OMEGA1"), *envOmega2 = getenv("LMSOR_OMEGA2");
	int gperiptosi;
	gperiptosi = build_omega(arxeio, re, cImags, cReals, min_w1, max_w1, min_w2, max_w2, epilogi, min_m_low, max_m_low, min_m_up, max_m_up,block_size-2,taskid,atoi(argv[7]));
	
	if( envOmega1 && envOmega2 ){
		cImags = cReals = 0;
		min_w1 = max_w1 = min_w2 = max_w2 = min_m_low = max_m_low = min_m_up = max_m_up = 0.0;
		double omega1 = atof(envOmega1), omega2 = atof(envOmega2);
		printf("\nNote: Read omega values from the environment (%.3f, %.3f)\n", omega1, omega2);
		min_w1 = max_w1 = omega1;
		min_w2 = max_w2 = omega2;
		for(i=0; i<=n+1; i++) {
		  for(j=0; j<=n+1; j++) {
		  	
			arr_w.set(i, j, ( (i+j)%2 == 0 ) ? omega1 : omega2);
			
		  }
		}
	}
 
#ifdef MODIFIED_SOR
	printf("R/B LMSOR method (5 point)\n");
#else
	printf("R/B LOCAL SOR method (5 point)\n");
#endif

//printf("!!!! n=%d\n", n);

#pragma omp parallel for private(j,x,y) num_threads(atoi(argv[7]))
    for(i=0; i<=n+1; i++) {
      x=i*h;
      for(j=0; j<=n+1; j++) {
        y=j*h;
		arr_u.set(i, j, initial_guess(x,y));
		
			/*if(taskid==0 && (omp_get_thread_num()==0)){
				fprintf(stderr, "arr_u[%d][%d]=%f\n",i,j,arr_u.get(i,j) );
			}*/
			
//printf("!!!%f,%f max=%f u=%f w=%f\n", x, y, max1, u.get(i, j), w.get(i, j));
/*        u0[i][j]=initial_guess(x,y);
        u1[i][j]=u0[i][j];*/
      }
    }
    metr=0;

	// Forward Reordering
	ts_start = getTimestamp();
/*	ro_u.copyfrom(arr_u);
#ifdef _PRECALC_
	ro_ffh.copyfrom(arr_ffh);
	ro_ggh.copyfrom(arr_ggh);
#else
	ro_l.copyfrom(arr_l);
	ro_r.copyfrom(arr_r);
	ro_t.copyfrom(arr_t);
	ro_b.copyfrom(arr_b);
#endif
	ro_w.copyfrom(arr_w);*/
	double reorderFwTime = getElapsedtime(ts_start)/1000.0f;

	ts_start = getTimestamp();
	int up,down,left,right;
	MPI_Cart_shift(new_comm, 0,1,&up,&down);
    MPI_Cart_shift(new_comm, 1,1,&left,&right);


	MPI_Request req_array_u[2][4];

#ifdef _PRECALC_
	MPI_Request req_array_ffh[2][4];
	MPI_Request req_array_ggh[2][4];
#else
	MPI_Request req_array_l[2][4];
	MPI_Request req_array_r[2][4];
	MPI_Request req_array_t[2][4];
	MPI_Request req_array_b[2][4];
#endif
	MPI_Request req_array_w[2][4];

int error_sum=0;
int it_counter=0;




	 



    int tag_array_u[] = {0,1,2,3};





	send_recv_init(arr_u._get_data(),up,down,left,right,new_comm,col_type,&req_array_u[0][0],tag_array_u,block_size);

#ifdef _PRECALC_
	int tag_array_ffh[] = {4,5,6,7};
	int tag_array_ggh[] = {8,9,10,11};
	send_recv_init(arr_ffh._get_data(),up,down,left,right,new_comm,col_type,&req_array_ffh[0][0],tag_array_ffh,block_size);
	send_recv_init(arr_ggh._get_data(),up,down,left,right,new_comm,col_type,&req_array_ggh[0][0],tag_array_ggh,block_size);
#else
	int tag_array_l[] = {4,5,6,7};
	int tag_array_r[] = {8,9,10,11};
	int tag_array_t[] = {12,13,14,15};
	int tag_array_b[] = {16,17,18,19};
	send_recv_init(arr_l._get_data(),up,down,left,right,new_comm,col_type,&req_array_l[0][0],tag_array_l,block_size);
	send_recv_init(arr_r._get_data(),up,down,left,right,new_comm,col_type,&req_array_r[0][0],tag_array_r,block_size);
	send_recv_init(arr_t._get_data(),up,down,left,right,new_comm,col_type,&req_array_t[0][0],tag_array_t,block_size);
	send_recv_init(arr_b._get_data(),up,down,left,right,new_comm,col_type,&req_array_b[0][0],tag_array_b,block_size);
#endif
	int tag_array_w[] = {20,21,22,23};
	send_recv_init(arr_w._get_data(),up,down,left,right,new_comm,col_type,&req_array_w[0][0],tag_array_w,block_size);

#ifdef _PRECALC_
	


	MPI_Start(&req_array_ffh[0][0]);
	MPI_Start(&req_array_ffh[0][1]);
	MPI_Start(&req_array_ffh[0][2]);
	MPI_Start(&req_array_ffh[0][3]);
	MPI_Start(&req_array_ffh[1][0]);
	MPI_Start(&req_array_ffh[1][1]);
	MPI_Start(&req_array_ffh[1][2]);
	MPI_Start(&req_array_ffh[1][3]);

	//my_start_all(&req_array_ffh[0][0]);

	MPI_Start(&req_array_ggh[0][0]);
	MPI_Start(&req_array_ggh[0][1]);
	MPI_Start(&req_array_ggh[0][2]);
	MPI_Start(&req_array_ggh[0][3]);
	MPI_Start(&req_array_ggh[1][0]);
	MPI_Start(&req_array_ggh[1][1]);
	MPI_Start(&req_array_ggh[1][2]);
	MPI_Start(&req_array_ggh[1][3]);

	//my_start_all(&req_array_ggh[0][0]);
	
#else

	MPI_Start(&req_array_l[0][0]);
	MPI_Start(&req_array_l[0][1]);
	MPI_Start(&req_array_l[0][2]);
	MPI_Start(&req_array_l[0][3]);
	MPI_Start(&req_array_l[1][0]);
	MPI_Start(&req_array_l[1][1]);
	MPI_Start(&req_array_l[1][2]);
	MPI_Start(&req_array_l[1][3]);

	//my_start_all(&req_array_l[0][0]);

	MPI_Start(&req_array_r[0][0]);
	MPI_Start(&req_array_r[0][1]);
	MPI_Start(&req_array_r[0][2]);
	MPI_Start(&req_array_r[0][3]);
	MPI_Start(&req_array_r[1][0]);
	MPI_Start(&req_array_r[1][1]);
	MPI_Start(&req_array_r[1][2]);
	MPI_Start(&req_array_r[1][3]);
	
	//my_start_all(&req_array_r[0][0]);

	MPI_Start(&req_array_t[0][0]);
	MPI_Start(&req_array_t[0][1]);
	MPI_Start(&req_array_t[0][2]);
	MPI_Start(&req_array_t[0][3]);
	MPI_Start(&req_array_t[1][0]);
	MPI_Start(&req_array_t[1][1]);
	MPI_Start(&req_array_t[1][2]);
	MPI_Start(&req_array_t[1][3]);

//	my_start_all(&req_array_t[0][0]);

	MPI_Start(&req_array_b[0][0]);
	MPI_Start(&req_array_b[0][1]);
	MPI_Start(&req_array_b[0][2]);
	MPI_Start(&req_array_b[0][3]);
	MPI_Start(&req_array_b[1][0]);
	MPI_Start(&req_array_b[1][1]);
	MPI_Start(&req_array_b[1][2]);
	MPI_Start(&req_array_b[1][3]);
	//my_start_all(&req_array_b[0][0]);
	
#endif

	MPI_Start(&req_array_w[0][0]);
	MPI_Start(&req_array_w[0][1]);
	MPI_Start(&req_array_w[0][2]);
	MPI_Start(&req_array_w[0][3]);
	MPI_Start(&req_array_w[1][0]);
	MPI_Start(&req_array_w[1][1]);
	MPI_Start(&req_array_w[1][2]);
	MPI_Start(&req_array_w[1][3]);

//	my_start_all(&req_array_w[0][0]);

#ifdef _PRECALC_

	MPI_Wait(&req_array_ffh[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ffh[1][3],MPI_STATUSES_IGNORE);

	//my_wait_all(&req_array_ffh[0][0]);

	MPI_Wait(&req_array_ggh[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_ggh[1][3],MPI_STATUSES_IGNORE);	
	//my_wait_all(&req_array_ggh[0][0]);

#else

	MPI_Wait(&req_array_l[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_l[1][3],MPI_STATUSES_IGNORE);

	//my_wait_all(&req_array_l[0][0]);

	MPI_Wait(&req_array_r[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_r[1][3],MPI_STATUSES_IGNORE);

	//my_wait_all(&req_array_r[0][0]);

	MPI_Wait(&req_array_t[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_t[1][3],MPI_STATUSES_IGNORE);

	//my_wait_all(&req_array_t[0][0]);	

	MPI_Wait(&req_array_b[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_b[1][3],MPI_STATUSES_IGNORE);

	//my_wait_all(&req_array_b[0][0]);

#endif

	MPI_Wait(&req_array_w[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_w[1][3],MPI_STATUSES_IGNORE);

	//my_wait_all(&req_array_w[0][0]);

#ifdef _PRECALC_
	
	ro_ffh.copyfrom(arr_ffh);
	
	ro_ggh.copyfrom(arr_ggh);

#else
	ro_l.copyfrom(arr_l);
	ro_r.copyfrom(arr_r);
	ro_t.copyfrom(arr_t);
	ro_b.copyfrom(arr_b);
#endif
	
	//ro_w.set(0,1,arr_w.get(0,1));
	
	ro_w.copyfrom(arr_w);
	
	MPI_Start(&req_array_u[0][0]);
	MPI_Start(&req_array_u[0][1]);
	MPI_Start(&req_array_u[0][2]);
	MPI_Start(&req_array_u[0][3]);
	MPI_Start(&req_array_u[1][0]);
	MPI_Start(&req_array_u[1][1]);
	MPI_Start(&req_array_u[1][2]);
	MPI_Start(&req_array_u[1][3]);


	

	MPI_Wait(&req_array_u[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&req_array_u[1][3],MPI_STATUSES_IGNORE);
	
	ro_u.copyfrom(arr_u);

	int tag_array_ro_u_red[] = {24,25,26,27};
	int tag_array_ro_u_black[] = {28,29,30,31};
 	
      do {
//	    timestamp ts_start;
	 //   fprintf(stderr,"MESA STI DO WHILE!!!!!!!!!!\n");


			/*if(taskid==3){
				fprintf(stderr, "EDW?????????????????\n" );
			}*/
          metr=metr+1;
//printf("Iteration: %ld, thread %d\n", metr, omp_get_thread_num());
//          ts_start = getTimestamp();
//		  max = fabs( ( 1. - w1[1][1] )*u[1][1] + w1[1][1]*( l[1][1]*u[1-1][1] + r[1][1]*u[1+1][1] + b[1][1]*u[1][1-1] + t[1][1]*u[1][1+1] ) );
		  sqrerror = 0.0;//fabs( ( 1. - w.get(1, 1) )*u.get(1, 1) + w.get(1, 1)*( l.get(1, 1)*u.get(1-1, 1) + r.get(1, 1)*u.get(1+1, 1) + b.get(1, 1)*u.get(1, 1-1) + t.get(1, 1)*u.get(1, 1+1) ) );
		

		double tsqrerror = 0.0;
        
		double* ro_u_pointer;

		
		ro_u_pointer = ro_u.get_pointer();
		
		
		MPI_Request red_reqs[2][4];
		MPI_Request black_reqs[2][4];




#pragma omp barrier
		if(k==1){
			calcSegment<0>(1, columns, 1, block_size-1, tsqrerror,block_size,taskid,metr);
		}
	MPI_Barrier(new_comm);	
	send_recv_init_rb(ro_u_pointer,up,down,left,right,new_comm,ro_col_type,&red_reqs[0][0],tag_array_ro_u_red,block_size);


	MPI_Start(&red_reqs[0][0]);
	MPI_Start(&red_reqs[0][1]);
	MPI_Start(&red_reqs[0][2]);
	MPI_Start(&red_reqs[0][3]);
	MPI_Start(&red_reqs[1][0]);
	MPI_Start(&red_reqs[1][1]);
	MPI_Start(&red_reqs[1][2]);
	MPI_Start(&red_reqs[1][3]);


	

	MPI_Wait(&red_reqs[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&red_reqs[1][3],MPI_STATUSES_IGNORE);

	MPI_Barrier(new_comm);

#pragma omp barrier
		if(k==1){
			calcSegment<1>(1, columns, 1, block_size-1, tsqrerror,block_size,taskid,metr);
		}

/*		for(i=start_pos_y; i<end_pos_y; i++){
		  for(j=start_pos_x; j<end_pos_x; j++){
            if ( (i+j)%2 != 0 ) {
				u.set(i, j, ( 1. - w.get(i, j) )*u.get(i, j) + w.get(i, j)*( l.get(i, j)*u.get(i-1, j) + r.get(i, j)*u.get(i+1, j) + b.get(i, j)*u.get(i, j-1) + t.get(i, j)*u.get(i, j+1) ) );
				double max1 = fabs(u.get(i, j));
              if( tmax<max1 ) tmax = max1;
            }
		  }
		}*/
	MPI_Barrier(new_comm);
	send_recv_init_rb(ro_u_pointer+(block_size*(columns+2)),up,down,left,right,new_comm,ro_col_type,&black_reqs[0][0],tag_array_ro_u_black,block_size);


	MPI_Start(&black_reqs[0][0]);
	MPI_Start(&black_reqs[0][1]);
	MPI_Start(&black_reqs[0][2]);
	MPI_Start(&black_reqs[0][3]);
	MPI_Start(&black_reqs[1][0]);
	MPI_Start(&black_reqs[1][1]);
	MPI_Start(&black_reqs[1][2]);
	MPI_Start(&black_reqs[1][3]);


	

	MPI_Wait(&black_reqs[0][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[0][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[0][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[0][3],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[1][0],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[1][1],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[1][2],MPI_STATUSES_IGNORE);
	MPI_Wait(&black_reqs[1][3],MPI_STATUSES_IGNORE);

	

	MPI_Barrier(new_comm);

//printf("\n%d. max=%4f\n", omp_get_thread_num(), tmax);
//			printf("final max set %f\n", tmax);
          	if(CHECK_SQRERROR){
				sqrerror += tsqrerror;
          	}
		  


			//printf("\nit=%4d max=%4f\n", metr, max);
		if(CHECK_SQRERROR){
	        if(k==1){ //no error
	          if (sqrerror <= e1){
	          	fprintf(stderr, "ERRRRRRRRRRRRRROOOOOOOOOOOOOOOOOOOOOOOOOOORRRRRRRRRRRRRRRRRRRRRRRRRR\n" );
	            k=0;
	           }
	          else {
	            if (isnan(sqrerror) || sqrerror >= 5.0e+8){
	              k=2;
	            }
	            else
	              k=1;
	          }
	        }
	     }
//          printf("(%f secs)\n", getElapsedtime(ts_start)/1000.0);



//printf("i'm thread %d with k=%d and metr=%d of %d\n", omp_get_thread_num(), k, metr, maxiter);
		int local_error;

		error_sum = 0;

		if(it_counter==ERROR_CHECK_ITERS){
			it_counter=0;

			if(k==1){ 		//no error
				local_error=0;
				MPI_Allreduce(&local_error, &error_sum, 1, MPI_INT, MPI_SUM, new_comm);
			}

			else{
				local_error=1;
				MPI_Allreduce(&local_error, &error_sum, 1, MPI_INT, MPI_SUM, new_comm);
			}

			
		}

		

		it_counter++;
		
		//end of omp master
/*#pragma omp_master			//////////debug
		{
			if(taskid==3){
				fprintf(stderr, "EFTASA EDW!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n" );
			}
		}*/

      } while ((error_sum!=numtasks) && (metr<=maxiter));


	
	
	 //END OF THREADSSSSSSSS
		
	#ifdef _PRECALC_
		ro_ffh.copyto(arr_ffh);
		ro_ggh.copyto(arr_ggh);
	#else
		ro_l.copyto(arr_l);
		ro_r.copyto(arr_r);
		ro_t.copyto(arr_t);
		ro_b.copyto(arr_b);
	#endif
		ro_w.copyto(arr_w);
	fprintf(stderr,"EFTASA TI EGINEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");
	double calc_time = getElapsedtime(ts_start)/1000.0f;
#ifdef _PRECALC_
	const double ACCESSES_PER_ELEMENT = 6.;
#else
	const double ACCESSES_PER_ELEMENT = 8.;
#endif
	printf(" %f seconds (%f milliseconds/iteration, %.2fGB/sec)\n", calc_time, 1000.0f*calc_time/metr, (ACCESSES_PER_ELEMENT*block_size*block_size*sizeof(double)*metr/calc_time)/(1024.*1024.*1024.));

	// Inverse Reordering
	ts_start = getTimestamp();
/*	ro_u.copyto(arr_u);
#ifdef _PRECALC_
	ro_ffh.copyto(arr_ffh);
	ro_ggh.copyto(arr_ggh);
#else
	ro_l.copyto(arr_l);
	ro_r.copyto(arr_r);
	ro_t.copyto(arr_t);
	ro_b.copyto(arr_b);
#endif
	ro_w.copyto(arr_w);*/
	double reorderInvTime = getElapsedtime(ts_start)/1000.0f;
	printf(" %f reordering overhead (%f forward reordering + %f inverse reordering)\n", reorderFwTime+reorderInvTime, reorderFwTime, reorderInvTime);
	printf(" %f seconds total time\n", reorderFwTime+reorderInvTime+calc_time);

#ifdef MODIFIED_SOR
	const char cMethod[] = "R/B LMSOR (CPU)";
#else
	const char cMethod[] = "R/B LOCAL SOR (CPU)";
#endif
#ifdef _OPENMP
	const char cKernel[] = "OpenMP";
#else
	const char cKernel[] = "Sequential";
#endif
	printf("___\nEXCEL friendly line follows:\n%d;%d;%.1f;%d;(%ld,%ld);%s;%s;%ld;%f;%f;%d\n___\n", block_size, epilogi, re, gperiptosi, cReals, cImags, cMethod, cKernel, metr, calc_time, reorderFwTime+reorderInvTime+calc_time, k!=2);

    if (k==2) {
      printf("oxi sygklisi ****");
      printf("\n periptosi : %3d\n", periptosi);
      fprintf(arxeio," oxi sygklisi  ****");
      fprintf(stderr,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NITER = %6ld   ",metr);
      exit(0);
    } else {
      printf("NITER = %6ld   ",metr);
      printf("\n periptosi : %3d\n", periptosi);
    }

    /////////////////////////////////1o fprintf//////////////////////////////
   
 /*  if(taskid != 0){
   		MPI_Request mpi_req;
   		double *arr_u_pointer = arr_u._get_data();
   		
   		MPI_Isend(arr_u_pointer+block_size+1,1,inner_block,0,taskid,new_comm,&mpi_req);
   		MPI_Wait(&mpi_req,MPI_STATUSES_IGNORE);
   		//fprintf(stderr, "eimai o taskid=%d kai TON ESTEILAAAAA\n",taskid );
   }

  // else{
   		MPI_Request* array_of_reqs = new MPI_Request[numtasks-1];
     	double *full_array = new double[NMAX*NMAX];

     	//int blocks_per_line  = sqrt(numtasks);
     	int ll=0;
     	int cc=0;
     	for(int iii = 1; iii < numtasks; iii++){
     		cc++;
 			if(cc*(block_size-2) < NMAX){
 				MPI_Irecv(full_array+(ll*NMAX)+(cc*(block_size-2)),1,block_type,iii,iii,new_comm,&(array_of_reqs[iii-1]));
 				MPI_Wait(&(array_of_reqs[iii-1]),MPI_STATUSES_IGNORE);
 			//	fprintf(stderr, "eimai o taskid=%d kai OUTE KANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN\n",taskid );

 			}
 			else{
 			//	fprintf(stderr, "EFTASA KAI DWWWWWWWWWWWW\n" );
 				if(ll+(block_size-2)  <  NMAX){
 					ll+=(block_size-2);
 					cc=0;
 					MPI_Irecv(full_array+(ll*NMAX)+(cc*(block_size-2)),1,block_type,iii,iii,new_comm,&(array_of_reqs[iii-1]));
 					MPI_Wait(&(array_of_reqs[iii-1]),MPI_STATUSES_IGNORE);
 					//fprintf(stderr, "????????????????????????????????????????????????????????\n");
 				}
 				else{
 					break;
 				}
 			}
     		
     		

     	}

     	for(int iii = 0; iii < block_size-2; iii++){
     			for(int jjj = 0; jjj < block_size-2; jjj++){
     				*(full_array+(iii*NMAX)+jjj) = arr_u.get(iii+1,jjj+1);
     			}
     		}*/

     	/*fprintf(stderr, "arr_u 200 267 = %f\n",arr_u.get(200,267) );
     	fprintf(stderr, "full array 199 266=%f\n",*(full_array+(199*NMAX)+266)  );

     	fprintf(stderr, "full array block_size-1 9+block_size-2 = %f\n", *(full_array+((block_size-1)*NMAX)+9) );*/

     	if (ektyp1==1) {
	      printf("RANGE_w1 = %5.4lf - %5.4lf \t RANGE_w2 = %5.4lf - %5.4lf\n", min_w1,max_w1,min_w2,max_w2);

	      printf("\n periptosi: %3d RANGE_m_low = %5.4lf - %5.4lf \t RANGE_m_up = %5.4lf - %5.4lf\n",
	      periptosi,min_m_low,max_m_low,min_m_up,max_m_up);

	      fprintf(arxeio,"\n  %5ld \t %5.4lf - %5.4lf \t %5.4lf - %5.4lf \n",
	      metr,min_m_low,max_m_low,min_m_up,max_m_up);

	      fprintf(arxeio,"\n  %3d \t %5.4lf - %5.4lf \t %5.4lf - %5.4lf \n",
	      periptosi,min_m_low,max_m_low,min_m_up,max_m_up);

	      fprintf(arxeio,"..............................................................\n");
	    }

	    if (ektyp2==1) {
	  /*    for(j=NMAX-1; j>=0; j--) {
	        for( i=0; i<=NMAX-1; i++) {
	//          fprintf(arxeio,"  %5.4e",u1[i][j]);
				fprintf(arxeio,"  %5.4e",*(full_array+(i*NMAX)+j));
	        }
	        fprintf(arxeio,"\n");
	      }*/
	      fprintf(arxeio," NITER = %6ld \t RANGE_w1 = %5.4lf - %5.4lf \t RANGE_w2 = %5.4lf - %5.4lf \n",
	              metr,min_w1,max_w1,min_w2,max_w2);
	      fprintf(arxeio," periptosi:  %3d   RANGE_m_low =  %5.4lf - %5.4lf \t RANGE_m_up= %5.4lf - %5.4lf \n",
	              periptosi,min_m_low,max_m_low,min_m_up,max_m_up);
	      fprintf(arxeio, "\n...............................................\n");
	    }

		fclose(arxeio);
  	
  // }

   MPI_Barrier(new_comm);

   

    
  if( argc < 2 )
    mypause();
  
  MPI_Finalize();

  return 0;
}


double initial_guess(double xx, double yy) {
  double u_arx;
  u_arx=xx*yy*(1.-xx)*(1.-yy);
  return u_arx;
}

double FF(int epilog, double re, double xx, double yy) {
  double f=0.0;
  switch (epilog) {
    case 1 :
      f = re*pow((2.*xx-10.),3.0);
      break;
    case 2 :
      f = re*(2.*xx-10.);
      break;
    case 3 :
      f = re*pow(10.,4.0);
      break;
    case 4 :
      f = re*pow((2.*xx-10.),5.0);
      break;
    case 5 :
      f = re*pow(xx,2.0);
      break;
    case 6 :
      f = re*(10.+pow(xx,2.0));
      break;
    case 7 :
      f = re*pow((10.+ pow(xx,2.0)),2.0);
      break;
    case 8 :
      f = re*pow((2*xx-1.),3.0);
      break;
    case 9 :
      f = re*pow(xx,2.0);
      break;
    case 10 :
/*          f = (1./2.)*re*(1.+pow(xx,2));*/
      f = re*(1.+pow(xx,2.0));
      break;
    case 11 :
      f = re*(1.-2.*xx);
      break;
    case 12 :
      f = re*(2.*xx-1.);
      break;
    case 13 :
      f = re*(10.-2*xx);
      break;
    case 14 :
      f = 0.;
      break;
    default :
      printf("sfalma ston syntelesth FF\n");
      break;
  }
  return (f);
}


double GG(int epilog, double re, double xx, double yy) {
  double g=0.0;
  switch (epilog) {
    case 1 :
      g = re*pow((2*yy-10.),3.0);
      break;
    case 2 :
      g = re*(2*yy-10.);
      break;
    case 3 :
      g = re*pow(10.,4.0);
      break;
    case 4 :
      g = re*pow((2.*yy-10.),5.0);
      break;
    case 5 :
      g = re*pow(xx,2.0);
      break;
    case 6 :
      g = re*(10.+pow(yy,2.0));
      break;
    case 7 :
      g = re*pow((10.+pow(yy,2.0)),2.0);
      break;
    case 8 :
      g = 0;
      break;
    case 9 :
      g =0;
      break;
    case 10 :
      g =100.;
      break;
    case 11 :
      g = re*(1.-2.*yy);
      break;
    case 12 :
      g = re*(2.*yy-1.);
      break;
    case 13 :
      g = re*(10.-2.*yy);
      break;
    case 14 :
      g = 0.;
      break;
    default :
      printf("sfalma ston syntelesth GG\n");
      break;
  }
  return (g);
}

void min_max_MAT( double MAT[], int n, double *min_MAT, double *max_MAT,int block_size ) {
  int i,j;
  
  *min_MAT=MAT[1*block_size+1];

  *max_MAT=MAT[1*block_size+1];

  for(i=1; i<=n; i++) {
    for(j=1; j<=n; j++) {
      if(*min_MAT > MAT[i*block_size+j]) {
         *min_MAT = MAT[i*block_size+j];
      }
      if(*max_MAT < MAT[i*block_size+j]) {
         *max_MAT = MAT[i*block_size+j];
      }
    }
  }
}





