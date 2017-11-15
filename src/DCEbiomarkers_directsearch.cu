#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "matrix.h"
#include "FFT_CODE/fft.h"
#include "FFT_CODE/complex.h"

#define NX 64
#define MAXIT 200
#define NPARAMS 3

template <typename T> void writeToBinary(const char *filename, T *data, int ndata, int *sizes, int dim);
template <typename T> void readFromBinary(const char *filename, T **data, int *ndata, int **sizes, int *dim);
template <typename T> __host__ __device__ T norm(T *vec, int n);
template <typename T> __host__ __device__ T SSQ(T *vec, int n);
__device__ void solveNLLSQ(float *J, float *dCt, float *dB, int npoints);
__device__ void gaussSolver(float *A,float *b, float *x);
__global__ void NLLSQkernel(int nsd, int nt, int *mriSizes, float difft, float *Ct, float *t, float *AIF, 
								complex *Cp_g, complex *fftCp_g, float *vp, float *ktrans, float *kep, float *converged,
								float *iter, float *res, int zlayer);

int main(void){
	size_t free,total;
  cudaError_t err = cudaSuccess;				// Error code to check return values for CUDA calls

	printf("---> Reading data...\n");
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
  float difft = (time[nt-1]-time[0])/(nt-1);	/* Average temporary step */

	/****************************************************/
  /* Reordering of C_t data. 													*/
	/* Data along time of each voxel will be continuous */
	/****************************************************/
  int aux_step = nVoxels;
  for(int i=0;i<mriSizes[0];i++){
    for(int j=0;j<mriSizes[1];j++){
      for(int k=0;k<mriSizes[2];k++){
        int aux_offset = k*mriSizes[0]*mriSizes[1] + j*mriSizes[0] + i;
        int data_offset = aux_offset*nt;
        Ct.getDataValues(aux_offset,aux_step,nt,Ct_aux,data_offset,1);
      }
    }
  }
	Ct.setData(Ct_aux);	/* Copy the C_t reordered to Ct */
  
  /* Test with any voxel */
/*  int px=248,py=275,pz=11;
  for(int i=0;i<mriSizes[3];i++){
    int idx= i*512*512*24 + (pz-1)*512*512 + (py-1)*512 + (px-1);
    printf("%f %f %f\n",time[i],AIF.data[i],Ct.data[idx]);
  }  */

	/******************************************************/ 
  /* Alloc memory for vp, ktrans, kep y other variables */
	/******************************************************/ 
	matrix<float> vp(3,mriSizes,nVoxels);						/* Vp matrix */
	matrix<float> ktrans(3,mriSizes,nVoxels);				/* Ktrans matrix */
	matrix<float> kep(3,mriSizes,nVoxels);					/* Kep matrix */
	matrix<float> converged(3,mriSizes,nVoxels);		/* Matrix of convergence of the NLLSQ method */
	matrix<float> iter(3,mriSizes,nVoxels);					/* Number of iterations per voxel */
	matrix<float> res(3,mriSizes,nVoxels);					/* Residuals per voxel */
	matrix<float> CtN(4,mriSizes,nVoxels*nt);				/* Numerical approximation of Ct */
  float *Cp_re = new float[NX];										/* AIF */
  complex *Cp = new complex[NX];									/* AIF (complex)*/
  complex *fftCp = new complex[NX];								/* fft(AIF) */
  for(int i=0;i<NX;i++)	Cp[i] = Cp_re[i] = 0.0;		/* Initialize to 0 all elements */
  for(int i=0;i<nt;i++)
    Cp[i] = Cp_re[i] = AIF.getDataValue(i);				/* Copy AIF to the real part of Cp */ 
  CFFT::Forward(Cp,fftCp,NX); 										/* FFT of Cp */
  for(int i=0;i<mriSizes[0];i++){
    for(int j=0;j<mriSizes[1];j++){
      for(int k=0;k<mriSizes[2];k++){
        int aux_offset = k*mriSizes[0]*mriSizes[1] + j*mriSizes[0] + i;
        res.setDataValue(aux_offset,1e10);
      }
    }
  }
/*  for(int i=0;i<NX;i++)  
    printf("%f %f %f %f\n",Cp[i].re(),Cp_re[i],fftCp[i].re(),fftCp[i].im()); */

	/*****************************/
	/* Allocate memory in device */
	/*****************************/
	if(cudaMemGetInfo(&free,&total) != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to get device memory info\n");
    return 0;
  }
	printf("\\--> Device total memory: %lluMB\n |-> Device free memory: %lluMB\n",(total/1024)/1024,(free/1024)/1024);
	printf("---> Allocating device memory...\n");
	int *dev_mriSizes;
	float *dev_time,*dev_AIF,*dev_Ct,*dev_vp,*dev_ktrans,*dev_kep,*dev_converged,*dev_res,*dev_iter;
	complex *dev_Cp,*dev_fftCp;
	err = cudaMalloc((void**)&dev_time, nt*sizeof(float));
	err = cudaMalloc((void**)&dev_AIF,nt*sizeof(float));
	err = cudaMalloc((void**)&dev_mriSizes,(nsd+1)*sizeof(int));
	err = cudaMalloc((void**)&dev_Ct, nVoxels*nt*sizeof(float));
	err = cudaMalloc((void**)&dev_vp, nVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_ktrans, nVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_kep, nVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_converged, nVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_iter, nVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_res, nVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_Cp, NX*sizeof(complex));
	err = cudaMalloc((void**)&dev_fftCp, NX*sizeof(complex));
	if ( err != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return 0;
  }
	if(cudaMemGetInfo(&free,&total) != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to get device memory info\n");
    return 0;
  }
	printf("\\--> Device total memory: %lluMB\n |-> Device free memory: %lluMB\n",(total/1024)/1024,(free/1024)/1024);

	/*********************************/
	/* Copy data from host to device */
	/*********************************/
	printf("---> Copying data to device...\n");
	err = cudaMemcpy(dev_time,time,nt*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_AIF,Cp_re,nt*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_mriSizes,mriSizes,(nsd+1)*sizeof(int),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_Ct,Ct.getDataPointer(),nVoxels*nt*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_vp,vp.getDataPointer(),nVoxels*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_ktrans,ktrans.getDataPointer(),nVoxels*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_kep,kep.getDataPointer(),nVoxels*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_converged,converged.getDataPointer(),nVoxels*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_res,res.getDataPointer(),nVoxels*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_Cp,Cp,NX*sizeof(complex),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_fftCp,fftCp,NX*sizeof(complex),cudaMemcpyHostToDevice);
	if ( err != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to copy data\n");
    return 0;
  }

	/********************/
	/* Kernel execution */
	/********************/
	float elapsed=0;
	cudaEvent_t start, stop;
	err = cudaEventCreate(&start);
	err = cudaEventCreate(&stop);
	err = cudaEventRecord(start, 0);
  printf("---> Kernel execution...\n");  
	dim3 block(32,32);
	dim3 grid(16,16,1);
	if(cudaThreadGetLimit(&total,cudaLimitStackSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	}
 	printf("\\--> Cuda Thread Limit Stack Size = %lluKB\n",total/1024);
	if(cudaThreadSetLimit(cudaLimitStackSize,8*total)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	} 
	if(cudaThreadGetLimit(&total,cudaLimitMallocHeapSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
 	printf(" |-> Cuda Thread Limit Malloc Heap Size = %lluMB\n",(total/1024)/1024);
	if(cudaThreadSetLimit(cudaLimitMallocHeapSize,8*total)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
	//for(int i=0;i<mriSizes[2]/grid.z;i++){
	for(int i=0;i<4096;i++){
		NLLSQkernel<<<grid,block>>>(nsd,nt,dev_mriSizes,difft,dev_Ct,dev_time,dev_AIF,
																dev_Cp,dev_fftCp,dev_vp,dev_ktrans,dev_kep,dev_converged,
																dev_iter,dev_res,i);
		if((i+1)%100 == 0){
			if( (err=cudaDeviceSynchronize()) != cudaSuccess){
				printf("err = %d\n",err);
 				fprintf(stderr, "Cuda error: Failed to synchronize\n");
 				return 0;
			}	
			err = cudaEventRecord(stop, 0);
			err = cudaEventSynchronize(stop);
			err = cudaEventElapsedTime(&elapsed, start, stop);
			printf(" |-> [NLLSQkernel  %d] The elapsed time in gpu was %.2f s.\n",i+1,elapsed/1000);
		}
	}
	if(cudaMemGetInfo(&free,&total) != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to get device memory info\n");
    return 0;
  }
	printf(" |-> Device total memory: %lluMB\n |-> Device free memory: %lluMB\n",(total/1024)/1024,(free/1024)/1024);
	if(cudaThreadGetLimit(&total,cudaLimitStackSize)){
  	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	}
 	printf(" |-> Cuda Thread Limit Stack Size = %lluKB\n",total/1024);
	if(cudaThreadGetLimit(&total,cudaLimitMallocHeapSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
 	printf(" |-> Cuda Thread Limit Malloc Heap Size = %lluMB\n",(total/1024)/1024);
	err = cudaEventRecord(stop, 0);
	err = cudaEventSynchronize(stop);
	err = cudaEventElapsedTime(&elapsed, start, stop);
	err = cudaEventDestroy(start);
	err = cudaEventDestroy(stop);
	printf(" |-> The elapsed time in gpu was %.2f s.\n", elapsed/1000);

	/*********************************/
	/* Copy data from device to host */
	/*********************************/
	printf("---> Copying data to host...\n");
	err = cudaMemcpy(vp.getDataPointer(),dev_vp,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
	err = cudaMemcpy(ktrans.getDataPointer(),dev_ktrans,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
	err = cudaMemcpy(kep.getDataPointer(),dev_kep,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
	err = cudaMemcpy(converged.getDataPointer(),dev_converged,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
	err = cudaMemcpy(iter.getDataPointer(),dev_iter,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
	err = cudaMemcpy(res.getDataPointer(),dev_res,nVoxels*sizeof(float),cudaMemcpyDeviceToHost);
	err = cudaMemcpy(CtN.getDataPointer(),dev_Ct,nVoxels*nt*sizeof(float),cudaMemcpyDeviceToHost);
	if ( err != cudaSuccess){ 
    fprintf(stderr, "Cuda error: Failed to copy data\n");
    return 0;
  }

	/**************************************************/
	/* Results may not be immediately available, 			*/
	/* so block device until all tasks have completed */
	/**************************************************/
	if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return 0;
  }

	/******************************************/
  /* Reordering of CtN data to DICOM format	*/
	/******************************************/
	int voxelsConverged=0;
	float global_res=0;
  aux_step = nVoxels;
  for(int i=0;i<mriSizes[0];i++){
    for(int j=0;j<mriSizes[1];j++){
      for(int k=0;k<mriSizes[2];k++){
        int aux_offset = k*mriSizes[0]*mriSizes[1] + j*mriSizes[0] + i;
        int data_offset = aux_offset*nt;
        CtN.getDataValues(data_offset,1,nt,Ct_aux,aux_offset,aux_step);
				global_res += res.getDataValue(aux_offset);
				voxelsConverged += converged.getDataValue(aux_offset);
      }
    }
  }
	CtN.setData(Ct_aux);	/* Copy the CtN */

	/******************/
	/* Save solutions */
	/******************/
	printf("---> Saving solution...\n");
	vp.writeToBinary("vp.dat");
  ktrans.writeToBinary("ktrans.dat");
  kep.writeToBinary("kep.dat");
  converged.writeToBinary("converged.dat");
  iter.writeToBinary("iterations.dat");
  res.writeToBinary("residuals.dat");
  CtN.writeToBinary("CtN.dat");
	printf("---> Global residual = %f\n",global_res);
	printf("---> %.3f%% voxels converged\n",(float)voxelsConverged/nVoxels*100);

	/**********************/
	/* Free device memory */
	/**********************/
	cudaFree(dev_time);
	cudaFree(dev_AIF);
	cudaFree(dev_mriSizes);
	cudaFree(dev_Ct);
	cudaFree(dev_vp);
	cudaFree(dev_ktrans);
	cudaFree(dev_kep);
	cudaFree(dev_converged);
	cudaFree(dev_Cp);
	cudaFree(dev_fftCp);

  printf("---> Finished!\n");
  return 0;
}

template <typename T>
__host__ __device__ T norm(T *vec, int n){
  T norm = 0;
  for(int i=0;i<n;i++)
    norm += vec[i]*vec[i];
  return sqrt(norm);
}

template <typename T>
__host__ __device__ T SSQ(T *vec, int n){
  T S = 0;
  for(int i=0;i<n;i++)
    S += vec[i]*vec[i];
  return S;
}

__device__ void solveNLLSQ(float *J, float *dCt, float *dB, int npoints){
	int J_offset,JT_offset;
  float JTxJ[NPARAMS][NPARAMS], JTxdCt[NPARAMS], aux;
  for(int i=0;i<NPARAMS;i++){
    JT_offset = i*npoints;
    for(int j=0;j<NPARAMS;j++){
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
/*	int px=threadIdx.x;
	if(px==0){
	for(int i=0;i<NPARAMS;i++){
		printf("|");
    for(int j=0;j<NPARAMS;j++)
			printf(" %f ",JTxJ[i][j]);
		printf(" | | %f |\n",JTxdCt[i]);
	}	} */
  gaussSolver(&JTxJ[0][0],JTxdCt,dB);
/*	if(px==0){
  for(int i=0;i<NPARAMS;i++)
		printf("| %f |\n",dB[i]);}	*/
}

__device__ void gaussSolver(float *A,float *b, float *x){
  int cont=0,pivot;
  float m[NPARAMS][NPARAMS],rhs[NPARAMS],aux,tol=1e-9;
  
  for(int i=0;i<NPARAMS;i++){
    for(int j=0;j<NPARAMS;j++)
      m[i][j] = A[cont++];
    rhs[i] = b[i];
  }
  
  for(int i=0;i<NPARAMS-1;i++){
    /* Pivoteo parcial */
    pivot=i;
    for(int j=i+1;j<NPARAMS;j++){
      if(fabs(m[j][i])>fabs(m[pivot][i]))
        pivot = j;
    }
    if(pivot!=i){
      for(int k=i;k<NPARAMS;k++){
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
				for(int k=0;k<NPARAMS-1;k++){
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
    for(int j=i+1;j<NPARAMS;j++){
      m[j][i] /= m[i][i];
      for(int k=i+1;k<NPARAMS;k++)
        m[j][k] -= m[j][i]*m[i][k];
      rhs[j] -= m[j][i]*rhs[i];
    }
  }
  
  /* Backward sustitution */
  for(int i=NPARAMS-1;i>=0;i--){
    x[i] = rhs[i];
    for(int j=i+1;j<NPARAMS;j++)
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

__global__ void NLLSQkernel(int nsd, int nt, int *mriSizes, float difft, float *Ct, float *t, float *AIF, 
								complex *Cp_g, complex *fftCp_g, float *vp, float *ktrans, float *kep, float *converged,
								float *iter, float *res, int zlayer){

	int px = threadIdx.x+blockDim.x*blockIdx.x;
	int py = threadIdx.y+blockDim.y*blockIdx.y;
	int pz = 10;
	if(px>=mriSizes[0]||py>=mriSizes[1]||pz>=mriSizes[2]) return;

  float *Ct_i = new float[nt];  						/* Ct data from 1 voxel */
  float *Ct_n = new float[nt];							/* Ct numerical approximation */
  float *dCt = new float[nt];								/* Diference between Ct_i and Ct_n */
  complex *Exp = new complex[NX];						/* Exponential function */
  complex *fftExp = new complex[NX];				/* fft(Exp) or fft(Exp.*t) */
  complex *convol = new complex[NX];				/* ifft(fftCp.*fftExp) */
  float *integral = new float[nt];					/* Truncated real part of convol using fft(Exp) */

	/*********************************************/
	/* ¿Conviene copiar los datos a cada thread? */
	/*********************************************/
	float *Cp_re = new float[nt];
	float *time = new float[nt];
	complex *fftCp = new complex[NX];
	for(int i=0;i<nt;i++){
		Cp_re[i] = AIF[i];
		time[i] = t[i];
	}
	for(int i=0;i<NX;i++){
		Exp[i] = 0.0;
		fftCp[i] = fftCp_g[i];	
	}

	/*********************************************/
	/*********************************************/
	/**********************/
  /* Initial parameters */
	/**********************/
  int param_offset = pz*mriSizes[0]*mriSizes[1] + py*mriSizes[0] + px;
  int data_offset = param_offset*nt;
  for(int l=0;l<nt;l++)
		Ct_i[l] = Ct[data_offset + l];								/* Ct data of voxel(px,py,pz) */
	float r;

	float vp0=(int)zlayer/256;//blockIdx.x*0.025;
	zlayer = zlayer%256;
	float ktrans0=(int)zlayer/16;//blockIdx.y*0.0005;
	float kep0=zlayer%16;//blockIdx.z*0.0005;
	vp0 *= 0.025;
	ktrans0 *= 0.0005;
	kep0 *= 0.0005;
//  for(float vp0=0;vp0<=0.5;vp0+=0.025){
//   for(float ktrans0=0;ktrans0<=0.01;ktrans0+=0.0005){
//    for(float kep0=0;kep0<=0.01;kep0+=0.0005){
	{{{
  		for(int l=0;l<nt;l++)
  			Exp[l] = exp(-kep0*time[l]);									/* Exponential function */
  		CFFT::Forward(Exp,fftExp,NX);										/* fft(Exp) */
      for(int l=0;l<NX;l++)
        fftExp[l] *= fftCp[l];												/* fftExp.*fftCp */
      CFFT::Inverse(fftExp,convol,NX);								/* ifft(fftCp.*fftExp) */
      for(int l=0;l<nt;l++){
        integral[l] = convol[l].re()*difft;						/* Truncated real part of convol using fft(Exp) */
        Ct_n[l] = vp0*Cp_re[l] + ktrans0*integral[l];	/* Ct numerical approximation */
        dCt[l] = Ct_i[l] - Ct_n[l];										/* Diference between Ct_i and Ct_n */
      }
      r = SSQ(dCt,nt);
			if(r<res[param_offset]){
  			vp[param_offset] = vp0;
  			ktrans[param_offset] = ktrans0;
  			kep[param_offset] = kep0;
  			res[param_offset] = r;
			}
		}
	 }
	}
//	printf("%d %d %d\n",px,py,pz);
//	printf("%d:%f\t%d:%f\t%d:%f\n",blockIdx.x,vp0,blockIdx.y,ktrans0,blockIdx.z,kep0);
//	printf("%f %f %f %f\n",r_min,beta_min[0],beta_min[1],beta_min[2]);
/*
	converged[param_offset] = 1;
 	for(int l=0;l<nt;l++)
 		Exp[l] = exp(-beta_min[2]*time[l]);
  CFFT::Forward(Exp,fftExp,NX);
  for(int l=0;l<NX;l++)
    fftExp[l] *= fftCp[l];
  CFFT::Inverse(fftExp,convol,NX);
  for(int l=0;l<nt;l++){
    integral[l] = convol[l].re()*difft;
    Ct_n[l] = beta_min[0]*Cp_re[l] + beta_min[1]*integral[l];
		Ct[data_offset+l] = Ct_n[l];
	}
*/
	if(param_offset>=mriSizes[0]*mriSizes[1]*mriSizes[2]){
		printf("param_offset = %d, max = %d",param_offset,mriSizes[0]*mriSizes[1]*mriSizes[2]);
		assert(0);
	}
	
  delete[] Ct_i;
  delete[] Ct_n;
  delete[] dCt;
  delete[] Exp;
  delete[] fftExp;
  delete[] convol;
  delete[] integral;
	delete[] Cp_re;
	delete[] time;
	delete[] fftCp;

	return;
}
