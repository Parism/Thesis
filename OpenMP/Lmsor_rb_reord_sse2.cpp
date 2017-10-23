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
#include <timestamp.h>
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

#ifndef NMAX
#define NMAX 1002
#endif
//#define NMAX 4002

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
#define _ALIGN_SZ_(x,y) ((x+y-1)/y)*y

#ifdef _OPENMP
#ifndef BLOCK_PARTITIONING
#error "Only block partitioning is implemented"
#endif
#endif

void mypause(void){
  printf("Press \"Enter\"\n");
  timestamp ts = getTimestamp();
  do {
    getchar();
  } while( getElapsedtime(ts)<100.0f );
}

long int pow(int b, int e){
	long int r = 1;
	for(int i=0; i<e; i++)
		r *= b;
	return r;
}

double pow2(double v){
	return v*v;
}

void validate_w(double w){
	if( isnan(w) ){
		printf("ERROR: Invalid w value\n");
		exit(1);
	}
}

template<int MAX>
class regular_array{
	double av[MAX][MAX];
public:
	inline void set(int row, int col, double v){
		av[row][col] = v;
	}
	inline double get(int row, int col){
		return av[row][col];
	}
/*	inline double* _get_data(void){
		return (double*)av;
	}*/
};

template<int MAX>
class redblack_array{
	double av[2][MAX][_ALIGN_SZ_(MAX/2, _DBLS_PER_CACHE_LINE_)];
public:
	inline void set(int row, int col, double v){
		if( (row+col)%2==0 )
			av[0][row][col/2] = v;
		else
			av[1][row][col/2] = v;
	}
	inline double get(int row, int col){
		if( (row+col)%2==0 )
			return av[0][row][col/2];
		else
			return av[1][row][col/2];
	}
	inline double* _get_data(int i){
		return (double*)av[i];
	}
	void copyfrom(regular_array<MAX> &arr){
		for(int iy=0; iy<MAX; iy++)
			for(int ix=0; ix<MAX; ix++)
				set(iy, ix, arr.get(iy, ix));
	}
	void copyto(regular_array<MAX> &arr){
		for(int iy=0; iy<MAX; iy++)
			for(int ix=0; ix<MAX; ix++)
				arr.set(iy, ix, get(iy, ix));
	}
};

regular_array<NMAX> arr_u;
#ifdef _PRECALC_
regular_array<NMAX> arr_ffh, arr_ggh;
#else
regular_array<NMAX> arr_l, arr_r, arr_t, arr_b;
#endif
regular_array<NMAX> arr_w;

redblack_array<NMAX> ro_u;
#ifdef _PRECALC_
redblack_array<NMAX> ro_ffh, ro_ggh;
#else
redblack_array<NMAX> ro_l, ro_r, ro_t, ro_b;
#endif
redblack_array<NMAX> ro_w;

double FF(int epil, double r1, double x1, double y1);
double GG(int epil, double r1, double x1, double y1);
double initial_guess(double x1, double y1);
void min_max_MAT(double MAT[], int n, double *min_MAT, double *max_MAT );

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
inline void calcSegment(const int bound_s_x, const int bound_e_x, const int bound_s_y, const int bound_e_y, double &sqrerror){
#ifdef _INTRINSIC_SSE2_

	const int pitch = _ALIGN_SZ_(NMAX/2, _DBLS_PER_CACHE_LINE_);
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
	int i,j=0;
#	pragma omp parallel for num_threads(2) schedule(dynamic,1) private(i,j)
	for(i=bound_s_y; i<bound_e_y; i++){
//#		pragma omp for
		for(j=bound_s_x; j<bound_e_x; j++){
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
}

// Zeroes all values in involved matrices by using the same thread assignment used during computation
// so memory is local to assigned processors on NUMA architectures
void firstTouch(void){
#pragma omp parallel
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
		int n=(NMAX-1);
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

int build_omega(FILE *arxeio, double re, long &cI, long &cR, double &min_w1, double &max_w1, double &min_w2, double &max_w2, int epilogi, double &min_m_low, double &max_m_low, double &min_m_up, double &max_m_up){
	double *w1 = new double[NMAX*NMAX];
	double *w2 = new double[NMAX*NMAX];
	double *m_low = new double[NMAX*NMAX];
	double *m_up = new double[NMAX*NMAX];
	
	int nn=NMAX-1, i, j, periptosi=0, tperiptosi=0;
	int gperiptosi=0;
	double x, y, C_E,C_W,C_N,C_S,g1,g2;
	long cImags = 0, cReals = 0;
	double pi=4.0*atan(1.);

	int n=nn-1;
    double h=1./(n+1.0);
    fprintf(arxeio, "\n h = %lf \t  n = %6d\n\n", h,n);
//    printf("Starting up...");
    timestamp ts_start = getTimestamp();
#pragma omp parallel for private(j,x,y,C_E,C_W,C_N,C_S,g1,g2) firstprivate(periptosi, tperiptosi) reduction(+ : cImags, cReals)
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
            m_up[i*NMAX+j] = 2.*(sqrt(C_E*C_W)+sqrt(C_N*C_S))*cos(pi*h);
            m_low[i*NMAX+j] = 2.*(sqrt(C_E*C_W)+sqrt(C_N*C_S))*cos(pi*(1.-h)/2.);
//printf("(%f) %f\n",2.*(sqrt(C_E*C_W)+sqrt(C_N*C_S)),m_low[i][j]);exit(1);
#ifdef MODIFIED_SOR
            w1[i*NMAX+j] = 2./(1.-m_up[i*NMAX+j]*m_low[i*NMAX+j]+sqrt((1.-m_up[i*NMAX+j])*(1.-m_low[i*NMAX+j])));
            w2[i*NMAX+j] = 2./(1.+m_up[i*NMAX+j]*m_low[i*NMAX+j]+sqrt((1.-m_up[i*NMAX+j])*(1.-m_low[i*NMAX+j])));
#else
			w1[i*NMAX+j] = w2[i*NMAX+j] = 2./(1.+sqrt(1.-m_up[i*NMAX+j]*m_up[i*NMAX+j]));
#endif
			cReals++;
            break;

          case 2:
            /* case 2 (Re(m)=0) imaginary */
            m_up[i*NMAX+j] = 2.*(sqrt(-C_E*C_W)+sqrt(-C_N*C_S))*cos(pi*h);
            m_low[i*NMAX+j] = 2.*(sqrt(-C_E*C_W)+sqrt(-C_N*C_S))*cos(pi*(1.-h)/2.);

#ifdef MODIFIED_SOR
            w1[i*NMAX+j] = 2./(1.-m_up[i*NMAX+j]*m_low[i*NMAX+j]+sqrt((1.+pow(m_up[i*NMAX+j],2.0))*(1.+pow(m_low[i*NMAX+j],2.0))));
            w2[i*NMAX+j] = 2./(1.+m_up[i*NMAX+j]*m_low[i*NMAX+j]+sqrt((1.+pow(m_up[i*NMAX+j],2.0))*(1.+pow(m_low[i*NMAX+j],2.0))));
#else
			w1[i*NMAX+j] = w2[i*NMAX+j] = 2./(1.+sqrt(1.+m_up[i*NMAX+j]*m_up[i*NMAX+j]));
#endif
			cImags++;
            break;

          case 3:
            /* case 3a  complex  */
            g1=pow((1.-pow((C_E+C_W), 2./3.)),-1./2.);
            w1[i*NMAX+j]=2./(1.+g1*fabs(C_N-C_S));
            break;

          case 4:
            /* case 3b complex  */
            g2=pow((1.-pow((C_N+C_S),2./3.)),-1./2.);
            w1[i*NMAX+j]=2./(1.+g2*fabs(C_E-C_W));
            w2[i*NMAX+j]=w1[i*NMAX+j];
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
		validate_w(w1[i*NMAX+j]);
		validate_w(w2[i*NMAX+j]);
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

	min_max_MAT(w1, n,  &min_w1, &max_w1);
    min_max_MAT(w2, n,  &min_w2, &max_w2);
    min_max_MAT(m_low, n,  &min_m_low, &max_m_low);
    min_max_MAT(m_up, n, &min_m_up, &max_m_up);

    printf("PDE :  %2d      h =  %lf     n = %6d \n",epilogi,h,n);
    printf("\n periptosi:  %2d RANGE_m_low = %5.4lf - %5.4lf \t RANGE_m_up = %5.4lf - %5.4lf\n",
           periptosi,min_m_low,max_m_low,min_m_up,max_m_up);

    fprintf(arxeio, "\n  %3d  \t  %5.4lf - %5.4lf \t %5.4lf - %5.4lf\n",
                    periptosi,min_m_low,max_m_low,min_m_up,max_m_up);

	// create unified matrix w
    for(i=0; i<=n+1; i++) {
      for(j=0; j<=n+1; j++) {
		arr_w.set(i,j, ( (i+j)%2 == 0 ) ? w1[i*NMAX+j] : w2[i*NMAX+j]);
//		printf("%d %d is %f\n", i,j,w[i][j]);
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
  int epilogi, periptosi=0, ektyp1, ektyp2;
  long maxiter;
  	int nn=NMAX-1;
	int n=nn-1;
//  double D;
  double min_w1, max_w1, min_w2, max_w2, min_m_low, max_m_low, min_m_up, max_m_up;
  double x, y;
    double h=1./(n+1.0);
//  double C_E, C_W, C_N, C_S, g1, g2;
  double e1=1.0e-6, sqrerror, re;
  char filename[20];
	timestamp ts_start;
  FILE *arxeio;

#ifdef _OPENMP
	firstTouch();
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

	// Support omega values input from the environment
	long cImags=0, cReals=0;
	char *envOmega1 = getenv("LMSOR_OMEGA1"), *envOmega2 = getenv("LMSOR_OMEGA2");
	int gperiptosi;
	gperiptosi = build_omega(arxeio, re, cImags, cReals, min_w1, max_w1, min_w2, max_w2, epilogi, min_m_low, max_m_low, min_m_up, max_m_up);
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

#pragma omp parallel for private(j,x,y)
    for(i=0; i<=n+1; i++) {
      x=i*h;
      for(j=0; j<=n+1; j++) {
        y=j*h;
		arr_u.set(i, j, initial_guess(x,y));
//printf("!!!%f,%f max=%f u=%f w=%f\n", x, y, max1, u.get(i, j), w.get(i, j));
/*        u0[i][j]=initial_guess(x,y);
        u1[i][j]=u0[i][j];*/
      }
    }
    metr=0;

	// Forward Reordering
	ts_start = getTimestamp();
	ro_u.copyfrom(arr_u);
#ifdef _PRECALC_
	ro_ffh.copyfrom(arr_ffh);
	ro_ggh.copyfrom(arr_ggh);
#else
	ro_l.copyfrom(arr_l);
	ro_r.copyfrom(arr_r);
	ro_t.copyfrom(arr_t);
	ro_b.copyfrom(arr_b);
#endif
	ro_w.copyfrom(arr_w);
	double reorderFwTime = getElapsedtime(ts_start)/1000.0f;

	ts_start = getTimestamp();
//#pragma omp parallel shared(sqrerror,metr,k) private(i)
   // {
//#pragma omp master
#ifdef _OPENMP
      printf("Using block partitioning (total threads %d)\n", omp_get_num_threads());
#endif
	  int num_threads = omp_get_num_threads();
	  fprintf(stderr, "NUMBER OF THREADS = %d\n",num_threads );
      int sy=num_threads, sx=1, sx_limit=(int)sqrt((double)sy);
//	  printf("limit %d\n", sx_limit);
      for(i=sx_limit; i>0; i--)
	if( num_threads%i==0 ){
	    sx = i;
	    sy = num_threads / i;
		break;
	  }
	  int bs_x=n/sx, bs_y=n/sy, lastoffset_x=n%sx, lastoffset_y=n%sy;
//	  printf("Thread %d dimensions %d x %d, data block %d x %d, last offset x=%d y=%d\n", omp_get_thread_num(), sx, sy, bs_x, bs_y, lastoffset_x, lastoffset_y);
	  int start_pos_x = 1+bs_x * (omp_get_thread_num()%sx),
		  start_pos_y = 1+bs_y * (omp_get_thread_num()/sx),
		  end_pos_x   = start_pos_x+bs_x+(omp_get_thread_num()%sx==sx-1?lastoffset_x:0),
		  end_pos_y   = start_pos_y+bs_y+(omp_get_thread_num()/sx==sy-1?lastoffset_y:0);
//      printf("Thread %d start pos %d %d, end pos %d x %d\n", omp_get_thread_num(), start_pos_x, start_pos_y, end_pos_x, end_pos_y);
//#pragma omp barrier
/*#pragma omp master
#ifdef _OPENMP
      printf("Using strip partitioning (total threads %d)\n", omp_get_num_threads());
#endif*/
      do {
//	    timestamp ts_start;
//#pragma omp barrier
//#pragma omp master
		//{
          metr=metr+1;
//printf("Iteration: %ld, thread %d\n", metr, omp_get_thread_num());
//          ts_start = getTimestamp();
//		  max = fabs( ( 1. - w1[1][1] )*u[1][1] + w1[1][1]*( l[1][1]*u[1-1][1] + r[1][1]*u[1+1][1] + b[1][1]*u[1][1-1] + t[1][1]*u[1][1+1] ) );
		  sqrerror = 0.0;//fabs( ( 1. - w.get(1, 1) )*u.get(1, 1) + w.get(1, 1)*( l.get(1, 1)*u.get(1-1, 1) + r.get(1, 1)*u.get(1+1, 1) + b.get(1, 1)*u.get(1, 1-1) + t.get(1, 1)*u.get(1, 1+1) ) );
		//}
//#pragma omp barrier
		double tsqrerror = 0.0;
        /* red/black ordering */
		calcSegment<0>(0, NMAX, 0, NMAX, tsqrerror);
/*		for(i=start_pos_y; i<end_pos_y; i++){
		  for(j=start_pos_x; j<end_pos_x; j++){
            if ( (i+j)%2 == 0 ) {
				u.set(i, j, ( 1. - w.get(i, j) )*u.get(i, j) + w.get(i, j)*( l.get(i, j)*u.get(i-1, j) + r.get(i, j)*u.get(i+1, j) + b.get(i, j)*u.get(i, j-1) + t.get(i, j)*u.get(i, j+1) ) );
				double max1 = fabs(u.get(i, j));
              if( tmax<max1 ) tmax = max1;
            }
		  }
		}*/
//#pragma omp barrier
		calcSegment<1>(0, NMAX, 0, NMAX, tsqrerror);
/*		for(i=start_pos_y; i<end_pos_y; i++){
		  for(j=start_pos_x; j<end_pos_x; j++){
            if ( (i+j)%2 != 0 ) {
				u.set(i, j, ( 1. - w.get(i, j) )*u.get(i, j) + w.get(i, j)*( l.get(i, j)*u.get(i-1, j) + r.get(i, j)*u.get(i+1, j) + b.get(i, j)*u.get(i, j-1) + t.get(i, j)*u.get(i, j+1) ) );
				double max1 = fabs(u.get(i, j));
              if( tmax<max1 ) tmax = max1;
            }
		  }
		}*/
//#pragma omp critical
         // {
//printf("\n%d. max=%4f\n", omp_get_thread_num(), tmax);
//			printf("final max set %f\n", tmax);
			sqrerror += tsqrerror;
		 // }
//#pragma omp barrier
//#pragma omp master
		//{
//printf("\nit=%4d max=%4f\n", metr, max);
          if (sqrerror <= e1)
            k=0;
          else {
            if (isnan(sqrerror) || sqrerror >= 5.0e+8)
              k=2;
            else
              k=1;
          }
//          printf("(%f secs)\n", getElapsedtime(ts_start)/1000.0);
		//}
//#pragma omp barrier
//printf("i'm thread %d with k=%d and metr=%d of %d\n", omp_get_thread_num(), k, metr, maxiter);
      } while ((k==1) && (metr<=maxiter));
//printf("end of %d\n", omp_get_thread_num());
	//}
	double calc_time = getElapsedtime(ts_start)/1000.0f;
#ifdef _PRECALC_
	const double ACCESSES_PER_ELEMENT = 6.;
#else
	const double ACCESSES_PER_ELEMENT = 8.;
#endif
	printf(" %f seconds (%f milliseconds/iteration, %.2fGB/sec)\n", calc_time, 1000.0f*calc_time/metr, (ACCESSES_PER_ELEMENT*NMAX*NMAX*sizeof(double)*metr/calc_time)/(1024.*1024.*1024.));

	// Inverse Reordering
	ts_start = getTimestamp();
	ro_u.copyto(arr_u);
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
	printf("___\nEXCEL friendly line follows:\n%d;%d;%.1f;%d;(%ld,%ld);%s;%s;%ld;%f;%f;%d\n___\n", NMAX, epilogi, re, gperiptosi, cReals, cImags, cMethod, cKernel, metr, calc_time, reorderFwTime+reorderInvTime+calc_time, k!=2);

    if (k==2) {
      printf("oxi sygklisi ****");
      printf("\n periptosi : %3d\n", periptosi);
      fprintf(arxeio," oxi sygklisi  ****");
      exit(0);
    } else {
      printf("NITER = %6ld   ",metr);
      printf("\n periptosi : %3d\n", periptosi);
    }

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
      for(j=n+1; j>=0; j--) {
        for( i=0; i<=n+1; i++) {
//          fprintf(arxeio,"  %5.4e",u1[i][j]);
			fprintf(arxeio,"  %5.4e",arr_u.get(i, j));
        }
        fprintf(arxeio,"\n");
      }
      fprintf(arxeio," NITER = %6ld \t RANGE_w1 = %5.4lf - %5.4lf \t RANGE_w2 = %5.4lf - %5.4lf \n",
              metr,min_w1,max_w1,min_w2,max_w2);
      fprintf(arxeio," periptosi:  %3d   RANGE_m_low =  %5.4lf - %5.4lf \t RANGE_m_up= %5.4lf - %5.4lf \n",
              periptosi,min_m_low,max_m_low,min_m_up,max_m_up);
      fprintf(arxeio, "\n...............................................\n");
    }

	fclose(arxeio);
  if( argc < 2 )
    mypause();
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

void min_max_MAT( double MAT[], int n, double *min_MAT, double *max_MAT ) {
  int i,j;

  *min_MAT=MAT[1*NMAX+1];

  *max_MAT=MAT[1*NMAX+1];

  for(i=1; i<=n; i++) {
    for(j=1; j<=n; j++) {
      if(*min_MAT > MAT[i*NMAX+j]) {
         *min_MAT = MAT[i*NMAX+j];
      }
      if(*max_MAT < MAT[i*NMAX+j]) {
         *max_MAT = MAT[i*NMAX+j];
      }
    }
  }
}
