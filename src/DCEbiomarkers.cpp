#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <time.h>
#include "FFT_CODE/fft.h"
#include "FFT_CODE/complex.h"

#define NX 64
#define MAXIT 1000

/* class matrix not implemented yet */
template <typename T>
class matrix{
	int ndata;
	int dim,*sizes;
	T 	*data;
};

template <typename T> void writeToBinary(const char *filename, T *data, int ndata, int *sizes, int dim);
template <typename T> void readFromBinary(const char *filename, T **data, int *ndata, int **sizes, int *dim);
template <typename T> T norm(T *vec, int n);
void solveNLLSQ(float *J, float *dCt, float *dB, int ndata, int nparams);
void solverGauss(float *A,float *b, float *x, int n);

int main(void){
  int ndata;
  int *Ct_sizes,Ct_dim,Ct_ndata;
  int *t_sizes,t_dim,t_ndata;
  int *AIF_sizes,AIF_dim,AIF_ndata=1;
  float *Ct_data,*Ct_aux,*AIF_data,*t_data;

  /* Leo AIF */
  readFromBinary("AIF.dat",&AIF_data,&AIF_ndata,&AIF_sizes,&AIF_dim);
  
  /* Leo t */
  readFromBinary("t.dat",&t_data,&t_ndata,&t_sizes,&t_dim);
  float difft = (t_data[t_ndata-1]-t_data[0])/(t_ndata-1);

  /* Leo C_t */
  readFromBinary("C_t.dat",&Ct_aux,&Ct_ndata,&Ct_sizes,&Ct_dim);
  Ct_data = new float[Ct_ndata];   /* Cambio el orden en que se guarda los datos en Ct; El tiempo pasa a ser el primer indice */
  int aux_step = Ct_sizes[0]*Ct_sizes[1]*Ct_sizes[2];
  for(int i=0;i<Ct_sizes[0];i++){
    for(int j=0;j<Ct_sizes[1];j++){
      for(int k=0;k<Ct_sizes[2];k++){
        int aux_offset = k*Ct_sizes[0]*Ct_sizes[1] + j*Ct_sizes[0] + i;
        int data_offset = aux_offset*t_ndata;
        for(int l=0;l<Ct_sizes[3];l++)
          Ct_data[data_offset+l] = Ct_aux[aux_offset + l*aux_step];
      }
    }
  }
  
  /* Prueba con un pixel cualquiera */
/*  int px=248,py=275,pz=11;
  for(int i=0;i<Ct_sizes[3];i++){
    int idx= i*512*512*24 + (pz-1)*512*512 + (py-1)*512 + (px-1);
    printf("%f %f %f\n",t_data[i],AIF_data[i],Ct_data[idx]);
  }  */

  /* Alloca memoria para vp, ktrans, kep y otros aux */
  ndata = Ct_sizes[0]*Ct_sizes[1]*Ct_sizes[2];
  float *vp = new float[ndata];
  float *ktrans = new float[ndata];
  float *kep = new float[ndata];
  float *J = new float[3*t_ndata];
  float *integral = new float[t_ndata];
  float *dintegral = new float[t_ndata];
  float *Ct = new float[t_ndata];
  float *Ct_k = new float[t_ndata];
  float *dCt = new float[t_ndata];
  float *converged = new float[ndata];
  complex *Cp = new complex[NX];
  complex *fftCp = new complex[NX];
  complex *Exp = new complex[NX];
  complex *fftExp = new complex[NX];
  complex *convol = new complex[NX];

  Cp = new complex[NX];
  for(int i=0;i<NX;i++){      /* Inicializo a 0 todas los elementos */
    Cp[i] = 0.0;  
    Exp[i] = 0.0;  
  }
  for(int i=0;i<t_ndata;i++)
    Cp[i] = AIF_data[i];      /* Copio AIF en la parte real de Cp */
  fftCp = new complex[NX];
  CFFT::Forward(Cp,fftCp,NX);  /* FFT de Cp */
/*  for(int i=0;i<NX;i++)  
    printf("%f %f %f\n",Cp[i].re(),fftCp[i].re(),fftCp[i].im());  */
  
  size_t t_i=clock();
  printf("Starting calculation...\n");  
	for(int i=0;i<Ct_sizes[0];i++){
 	//for(int i=213;i<247;i++){
    int maxit_step = 0;
    for(int j=0;j<Ct_sizes[1];j++){
    //for(int j=239;j<278;j++){
      for(int k=0;k<Ct_sizes[2];k++){
      //for(int k=10;k<11;k++){
				//printf("Voxel(%d,%d,%d)\n",i,j,k);
        /* Parámetros iniciales */ 
        int param_offset = k*Ct_sizes[0]*Ct_sizes[1] + j*Ct_sizes[0] + i;
        int data_offset = param_offset*t_ndata;
        float vp0=0.1,ktrans0=0.3/60,kep0=0.3/60;
        converged[param_offset] = 0;
        for(int l=0;l<Ct_sizes[3];l++){
          Ct[l] = Ct_data[data_offset + l];
          Exp[l] = exp(-kep0*t_data[l]);
					//printf("%f %f %f %f\n",Ct[l],Exp[l].re());
        }
        CFFT::Forward(Exp,fftExp,NX);
        for(int l=0;l<NX;l++)
          fftExp[l] *= fftCp[l];
        CFFT::Inverse(fftExp,convol,NX);
        for(int l=0;l<Ct_sizes[3];l++){
          integral[l] = convol[l].re()*difft;
          Ct_k[l] = vp0*AIF_data[l] + ktrans0*integral[l];
          dCt[l] = Ct[l] - Ct_k[l];
        }
        float r_old = norm(dCt,t_ndata);
        
        /* Cuadrados mínimos no-lineal */
  			int its = 0;
				int nparams = 3;
        float dB[3]={1e10,1e10,1e10},beta[3],beta_min[3];
        float dr=1e10;
        float r_min=1e10,r_new;
        while( its<MAXIT && (norm(dB,3)>1e-6||dr>1e-3) && kep0>1e-9 ){        
          its++;
          for(int l=0;l<Ct_sizes[3];l++)
            Exp[l] *= t_data[l];
          CFFT::Forward(Exp,fftExp,NX);
          for(int l=0;l<NX;l++)
            fftExp[l] *= fftCp[l];
          CFFT::Inverse(fftExp,convol,NX);
          for(int l=0;l<Ct_sizes[3];l++){
            dintegral[l] = convol[l].re()*difft;
            J[l] = AIF_data[l];    
            J[t_ndata+l] = integral[l];
            J[2*t_ndata+l] = -ktrans0*dintegral[l];
						//if(its==1)
							//printf("%f %f %f\n",J[l],J[t_ndata+l],J[2*t_ndata+l]);
          }
          solveNLLSQ(J,dCt,dB,t_ndata,nparams);
          if((vp0+dB[0])<0)
            dB[0] = -vp0*3/4;      
          if((ktrans0+dB[1])<0)
            dB[1] = -ktrans0*3/4;      
          if((kep0+dB[2])<0)
            dB[2] = -kep0*3/4;      
          vp0 += dB[0];
          ktrans0 += dB[1];
          kep0 += dB[2];
          beta[0] = vp0;
          beta[1] = ktrans0;
          beta[2] = kep0;
					//printf("%f,%f,%f\n",beta[0],beta[1],beta[2]);
          for(int l=0;l<Ct_sizes[3];l++)
            Exp[l] = exp(-kep0*t_data[l]);
          CFFT::Forward(Exp,fftExp,NX);
          for(int l=0;l<NX;l++)
            fftExp[l] *= fftCp[l];
          CFFT::Inverse(fftExp,convol,NX);
          for(int l=0;l<Ct_sizes[3];l++){
            integral[l] = convol[l].re()*difft;
            Ct_k[l] = vp0*AIF_data[l] + ktrans0*integral[l];
            dCt[l] = Ct[l] - Ct_k[l];
          }
          r_new = norm(dCt,t_ndata);
          if(r_new<r_min){
            r_min=r_new;
            beta_min[0] = beta[0];
            beta_min[1] = beta[1];
            beta_min[2] = beta[2];
          }  
          dr = fabs(r_new-r_old)/r_new;
					//printf("%f %f %f %f %f %f\n",dr,r_new,norm(dB,nparams),beta[0],beta[1],beta[2]);
					//getchar();
          r_old = r_new;
        }
        if(its!=MAXIT)
          converged[param_offset] = 1;
        vp[param_offset] = beta_min[0];
        ktrans[param_offset] = beta_min[1];
        kep[param_offset] = beta_min[2];
        if(its>maxit_step && its<MAXIT)
          maxit_step = its;
      }
    }  
		if((i+1)%50==0){
	    writeToBinary("vp.dat",vp,ndata,Ct_sizes,3);
  	  writeToBinary("ktrans.dat",ktrans,ndata,Ct_sizes,3);
    	writeToBinary("kep.dat",kep,ndata,Ct_sizes,3);
    	writeToBinary("converged.dat",converged,ndata,Ct_sizes,3);
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

void solveNLLSQ(float *J, float *dCt, float *dB, int ndata, int nparams){
	int J_offset,JT_offset;
  float JTxJ[nparams][nparams], JTxdCt[nparams], aux;
  for(int i=0;i<nparams;i++){
    JT_offset = i*ndata;
    for(int j=0;j<nparams;j++){
      aux=0;
      J_offset = j*ndata;
      for(int k=0;k<ndata;k++)
        aux += J[JT_offset+k] * J[J_offset+k];
      JTxJ[i][j] = aux;
    }
    aux = 0;
    for(int k=0;k<ndata;k++)
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
  solverGauss(&JTxJ[0][0],JTxdCt,dB,nparams);
/*  for(int i=0;i<nparams;i++)
		printf("| %f |\n",dB[i]);	*/
}

void solverGauss(float *A,float *b, float *x, int n){
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

/* Replaced by a template */
/*
void writeToBinary(const char *filename, float *data, int ndata, int *sizes, int dim);
void writeToBinary(const char *filename, float *data, int ndata, int *sizes, int dim){
  FILE *fid;
  fid = fopen(filename,"wb"); 
  assert(fid);
  fwrite(&dim,sizeof(int),1,fid);
  fwrite(sizes,sizeof(int),dim,fid);
  fwrite(data,sizeof(float),ndata,fid);  
  fclose(fid);
}	*/

/* Replaced by a template */
/*
void readFromBinary(const char *filename, float **data, int *ndata, int **sizes, int *dim);
void readFromBinary(const char *filename, float **data, int *ndata, int **sizes, int *dim){
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
  *data = new float[*ndata];
  assert(*data);
  fread(*data,sizeof(float),*ndata,fid);  
  fclose(fid);
}	*/
