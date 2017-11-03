#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "matrix.h"
#include "FFT_CODE/fft.h"
#include "FFT_CODE/complex.h"

#define NX 64
#define MAXIT 1000
#define NPARAMS 3

template <typename T> void writeToBinary(const char *filename, T *data, int ndata, int *sizes, int dim);
template <typename T> void readFromBinary(const char *filename, T **data, int *ndata, int **sizes, int *dim);
template <typename T> T norm(T *vec, int n);
void solveNLLSQ(float *J, float *dCt, float *dB, int npoints, int nparams);
void gaussSolver(float *A,float *b, float *x, int n);

int main(void){
	matrix<float> t("t.dat");							/* Read t */
  int nt = t.getNumData();							/* Size of temporal dimension */
  float *time = t.getData();						/* Time data */
	matrix<float> AIF("AIF.dat");					/* Read AIF */
	matrix<float> Ct("C_t.dat"); 					/* Read C_t but still needs a data reordering */
	int nsd = Ct.getDim()-1;							/* Number of spatial dimensions */
	int *mriSizes = Ct.getSizes();				/* MRI Sizes */ 
	int nVoxels = 1;
	for(int i=0;i<nsd;i++)
  	nVoxels *= mriSizes[i];							/* Number of MRI voxels */
  float *Ct_aux = new float[nVoxels*nt];/* Here will be reordered C_t */

	/* Average temporary step */
  float difft = (time[nt-1]-time[0])/(nt-1);	

  /* Reordering of C_t data. Data along time of each voxel will be continuous */
  int aux_step = nVoxels;
  for(int i=0;i<mriSizes[0];i++){
    for(int j=0;j<mriSizes[1];j++){
      for(int k=0;k<mriSizes[2];k++){
        int aux_offset = k*mriSizes[0]*mriSizes[1] + j*mriSizes[0] + i;
        int data_offset = aux_offset*nt;
        Ct.getDataValues(aux_offset,aux_step,nt,Ct_aux,data_offset,1);
//        for(int l=0;l<mriSizes[3];l++)
//          Ct.setDataValue(data_offset+l,Ct_aux.getDataValue(aux_offset + l*aux_step));
      }
    }
  }
	Ct.setData(Ct_aux);	/* Copy the C_t reordered to Ct */
	delete[] Ct_aux;
  
  /* Test with any voxel */
/*  int px=248,py=275,pz=11;
  for(int i=0;i<mriSizes[3];i++){
    int idx= i*512*512*24 + (pz-1)*512*512 + (py-1)*512 + (px-1);
    printf("%f %f %f\n",time[i],AIF.data[i],Ct.data[idx]);
  }  */

  /* Alloc memory for vp, ktrans, kep y other variables */
	matrix<float> vp(3,mriSizes,nVoxels);					/* Vp matrix */
	matrix<float> ktrans(3,mriSizes,nVoxels);			/* Ktrans matrix */
	matrix<float> kep(3,mriSizes,nVoxels);				/* Kep matrix */
	matrix<float> converged(3,mriSizes,nVoxels);	/* Matrix of convergence of the NLLSQ method */
  float *J = new float[NPARAMS*nt];					/* Jacobian for the NLLSQ method */
  float *Ct_i = new float[nt];  						/* Ct data from 1 voxel */
  float *Ct_n = new float[nt];							/* Ct numerical approximation */
  float *dCt = new float[nt];								/* Diference between Ct_i and Ct_n */
  float *Cp_re = new float[NX];							/* AIF */
  complex *Cp = new complex[NX];						/* AIF (complex)*/
  complex *fftCp = new complex[NX];					/* fft(AIF) */
  complex *Exp = new complex[NX];						/* Exponential function */
  complex *fftExp = new complex[NX];				/* fft(Exp) or fft(Exp.*t) */
  complex *convol = new complex[NX];				/* ifft(fftCp.*fftExp) */
  float *integral = new float[nt];					/* Truncated real part of convol using fft(Exp) */
  float *dintegral = new float[nt];					/* Truncated real part of convol using fft(Exp.*t) */

  for(int i=0;i<NX;i++){      /* Initialize to 0 all elements */
    Cp[i] = Cp_re[i] = 0.0;  
    Exp[i] = 0.0;  
  }
  for(int i=0;i<nt;i++)    		/* Copy AIF to the real part of Cp */
    Cp[i] = Cp_re[i] = AIF.getDataValue(i); 
  CFFT::Forward(Cp,fftCp,NX); /* FFT of Cp */
/*  for(int i=0;i<NX;i++)  
    printf("%f %f %f %f\n",Cp[i].re(),Cp_re[i],fftCp[i].re(),fftCp[i].im()); */

  size_t t_i=clock();
  printf("Starting calculation...\n");  
	for(int i=0;i<mriSizes[0];i++){
 	//for(int i=213;i<247;i++){
    int maxit_step = 0;								/* Maximum iteration number needed for convergence in a step 'i' */
    for(int j=0;j<mriSizes[1];j++){
    //for(int j=239;j<278;j++){
      for(int k=0;k<mriSizes[2];k++){
      //for(int k=10;k<11;k++){
				//printf("Voxel(%d,%d,%d)\n",i,j,k);
        /* Initial parameters */
        int param_offset = k*mriSizes[0]*mriSizes[1] + j*mriSizes[0] + i;
        int data_offset = param_offset*nt;
        float vp0=0.1,ktrans0=0.3/60,kep0=0.3/60;
        Ct.getDataValues(data_offset,1,mriSizes[3],Ct_i,0,1);		/* Ct data of voxel(i,j,k) */
        for(int l=0;l<mriSizes[3];l++)
          Exp[l] = exp(-kep0*time[l]);				/* Exponential function */
					//printf("%f %f %f %f\n",Ct_i[l],Exp[l].re());

        CFFT::Forward(Exp,fftExp,NX);						/* fft(Exp) */
        for(int l=0;l<NX;l++)
          fftExp[l] *= fftCp[l];								/* fftExp.*fftCp */
        CFFT::Inverse(fftExp,convol,NX);				/* ifft(fftCp.*fftExp) */
        for(int l=0;l<mriSizes[3];l++){
          integral[l] = convol[l].re()*difft;		/* Truncated real part of convol using fft(Exp) */
          Ct_n[l] = vp0*Cp_re[l] + ktrans0*integral[l];	/* Ct numerical approximation */
          dCt[l] = Ct_i[l] - Ct_n[l];						/* Diference between Ct_i and Ct_n */
        }
        float r_old = norm(dCt,nt);
  
        /* Non-Liniear Least Square Method */
  			int its = 0;
        float dB[NPARAMS]={1e10,1e10,1e10};
				float beta[NPARAMS]={vp0,ktrans0,kep0};  		/* SHOULD CHANGE IF NPARAMS CHANGE */
				float beta_min[NPARAMS]={vp0,ktrans0,kep0};	/* SHOULD CHANGE IF NPARAMS CHANGE */
        float dr=1e10;
        float r_min=1e10,r_new;
        while( its<MAXIT && (norm(dB,NPARAMS)>1e-6||dr>1e-3) && kep0>1e-9 ){        
          its++;
          for(int l=0;l<mriSizes[3];l++)
            Exp[l] *= time[l];
          CFFT::Forward(Exp,fftExp,NX);
          for(int l=0;l<NX;l++)
            fftExp[l] *= fftCp[l];
          CFFT::Inverse(fftExp,convol,NX);
          for(int l=0;l<mriSizes[3];l++){
            dintegral[l] = convol[l].re()*difft;
            J[l] = Cp_re[l];    
            J[nt+l] = integral[l];
            J[2*nt+l] = -ktrans0*dintegral[l];
						//if(its==1)
							//printf("%f %f %f %f\n",J[l],J[nt+l],J[2*nt+l],dCt[l]);
          }
          solveNLLSQ(J,dCt,dB,nt,NPARAMS);
					for(int i=0;i<NPARAMS;i++){
          	if((beta[i]+dB[i])<0)
            	dB[i] = -3.0/4.0*beta[i];      
						beta[i] += dB[i];
					} 
          vp0 = beta[0];
          ktrans0 = beta[1];
          kep0 = beta[2];
	
          for(int l=0;l<mriSizes[3];l++)
            Exp[l] = exp(-kep0*time[l]);
          CFFT::Forward(Exp,fftExp,NX);
          for(int l=0;l<NX;l++)
            fftExp[l] *= fftCp[l];
          CFFT::Inverse(fftExp,convol,NX);
          for(int l=0;l<mriSizes[3];l++){
            integral[l] = convol[l].re()*difft;
            Ct_n[l] = vp0*Cp_re[l] + ktrans0*integral[l];
            dCt[l] = Ct_i[l] - Ct_n[l];
          }
          r_new = norm(dCt,nt);
          if(r_new<r_min){
            r_min=r_new;
						for(int i=0;i<NPARAMS;i++)
            	beta_min[i] = beta[i];
          }  
          dr = fabs(r_new-r_old)/r_new;
					//printf("%f %f %f %f %f %f\n",dr,r_new,norm(dB,NPARAMS),beta[0],beta[1],beta[2]);
					//getchar();
          r_old = r_new;
        }
        if(its!=MAXIT)	
					converged.setDataValue(param_offset,1);
        vp.setDataValue(param_offset,beta_min[0]);
        ktrans.setDataValue(param_offset,beta_min[1]);
        kep.setDataValue(param_offset,beta_min[2]);
        if(its>maxit_step && its<MAXIT)
          maxit_step = its;
      }
    }  
		if((i+1)%50==0){
	    vp.writeToBinary("vp.dat");
  	  ktrans.writeToBinary("ktrans.dat");
    	kep.writeToBinary("kep.dat");
    	converged.writeToBinary("converged.dat");
		}
    printf("Voxel(%d,:,:) -> ",i+1);
    printf("max.its = %4d -> ",maxit_step);
		printf("%f seconds\n",(double)(clock()-t_i)/CLOCKS_PER_SEC);
    fflush(stdout);
  }
  printf("Finished in %f seconds...\n",(double)(clock()-t_i)/CLOCKS_PER_SEC);
  return 0;
}

template <typename T>
T norm(T *vec, int n){
  T norm = 0;
  for(int i=0;i<n;i++)
    norm += vec[i]*vec[i];
  return sqrt(norm);
}

void solveNLLSQ(float *J, float *dCt, float *dB, int npoints, int nparams){
	int J_offset,JT_offset;
  float JTxJ[nparams][nparams], JTxdCt[nparams], aux;
  for(int i=0;i<nparams;i++){
    JT_offset = i*npoints;
    for(int j=0;j<nparams;j++){
      aux=0;
      J_offset = j*npoints;
      for(int k=0;k<npoints;k++)
        aux += J[JT_offset+k] * J[J_offset+k];
      JTxJ[i][j] = aux;
    }
    aux = 0;
    for(int k=0;k<npoints;k++)
      aux += J[JT_offset+k] * dCt[k];
    JTxdCt[i] = aux;
  }
	/* Write matrix and rhs */
/*	for(int i=0;i<nparams;i++){
		printf("|");
    for(int j=0;j<nparams;j++)
			printf(" %f ",JTxJ[i][j]);
		printf(" | | %f |\n",JTxdCt[i]);
	}	*/
  gaussSolver(&JTxJ[0][0],JTxdCt,dB,nparams);
/*  for(int i=0;i<nparams;i++)
		printf("| %f |\n",dB[i]);	*/
}

void gaussSolver(float *A,float *b, float *x, int n){
  int cont=0,pivot;
  float m[n][n],rhs[n],aux,tol=1e-9;
  
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++)
      m[i][j] = A[cont++];
    rhs[i] = b[i];
  }
  
  for(int i=0;i<n-1;i++){
    /* Pivoteo parcial */
    pivot=i;
    for(int j=i+1;j<n;j++){
      if(fabs(m[j][i])>fabs(m[pivot][i]))
        pivot = j;
    }
    if(pivot!=i){
      for(int k=i;k<n;k++){
        aux = m[i][k];
        m[i][k] = m[pivot][k];
        m[pivot][k] = aux;
      }
      aux = rhs[i];
      rhs[i] = rhs[pivot];
      rhs[pivot] = aux;
    }
    if(fabs(m[i][i])<tol){
			if(i!=0){		/* Arreglo casero cuando kep muy grande */
				for(int k=0;k<n-1;k++){
					m[k][i]=0;			
				}
				m[i][i]=1;
			}
			else{
      	printf("Pivot ~ 0 en solver3x3().");
      	assert(0);
			}
    }
    
    /* Reducción */
    for(int j=i+1;j<n;j++){
      m[j][i] /= m[i][i];
      for(int k=i+1;k<n;k++)
        m[j][k] -= m[j][i]*m[i][k];
      rhs[j] -= m[j][i]*rhs[i];
    }
  }
  
  /* Backward sustitution */
  for(int i=n-1;i>=0;i--){
    x[i] = rhs[i];
    for(int j=i+1;j<n;j++)
      x[i] -= m[i][j]*x[j];
    x[i] /= m[i][i];
  }
}

template <typename T> 
void writeToBinary(const char *filename, T *data, int ndata, int *sizes, int dim){
  FILE *fid;
  fid = fopen(filename,"wb"); 
  assert(fid);
  fwrite(&dim,sizeof(int),1,fid);
  fwrite(sizes,sizeof(int),dim,fid);
  fwrite(data,sizeof(T),ndata,fid);  
  fclose(fid);
}

template <typename T> 
void readFromBinary(const char *filename, T **data, int *ndata, int **sizes, int *dim){
  FILE *fid;
  fid = fopen(filename,"rb"); 
  assert(fid);
  fread(dim,sizeof(int),1,fid);
  *sizes = new int[*dim];
  assert(*sizes);
  fread(*sizes,sizeof(int),*dim,fid);
  *ndata = 1;
  for(int i=0;i<*dim;i++)
    *ndata*=(*sizes)[i];
  *data = new T[*ndata];
  assert(*data);
  fread(*data,sizeof(T),*ndata,fid);  
  fclose(fid);
}
