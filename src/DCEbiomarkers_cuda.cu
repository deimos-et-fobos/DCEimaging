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
	float mainTime=0;
  cudaError_t err = cudaSuccess;				// Error code to check return values for CUDA calls
	size_t free,total,heapSize,stackSize;
	cudaEvent_t mainStart, mainStop;
	err = cudaEventCreate(&mainStart);
	err = cudaEventCreate(&mainStop);
	err = cudaEventRecord(mainStart, 0);

	printf("* Reading data...\n");
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
 	int nPixels = nVoxels/mriSizes[nsd-1];/* Number of pixels per MRI Slice */
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
	matrix<float> ve(3,mriSizes,nVoxels);						/* Ve matrix */
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
/*  for(int i=0;i<NX;i++)  
    printf("%f %f %f %f\n",Cp[i].re(),Cp_re[i],fftCp[i].re(),fftCp[i].im()); */

	/*****************************/
	/* Allocate memory in device */
	/*****************************/
	if(cudaMemGetInfo(&free,&total) != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to get device memory info\n");
    return 0;
  }
	printf("|\\\n");
	printf("|* Device total memory: %lluMB\n|* Device free memory: %lluMB\n",(total/1024)/1024,(free/1024)/1024);
	if(cudaThreadGetLimit(&stackSize,cudaLimitStackSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	}
 	printf("|* Cuda Thread Limit Stack Size = %lluKB\n",stackSize/1024);
	if(cudaThreadGetLimit(&heapSize,cudaLimitMallocHeapSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
 	printf("|* Cuda Thread Limit Malloc Heap Size = %lluMB\n",(heapSize/1024)/1024);
 	printf("|/\n");
	printf("* Allocating device memory...\n");
	int *dev_mriSizes;
	float *dev_time,*dev_AIF,*dev_Ct,*dev_vp,*dev_ktrans,*dev_kep,*dev_converged,*dev_res,*dev_iter;
	complex *dev_Cp,*dev_fftCp;
	err = cudaMalloc((void**)&dev_time, nt*sizeof(float));
	err = cudaMalloc((void**)&dev_AIF,nt*sizeof(float));
	err = cudaMalloc((void**)&dev_mriSizes,(nsd+1)*sizeof(int));
	err = cudaMalloc((void**)&dev_Ct, nPixels*nt*sizeof(float));
	err = cudaMalloc((void**)&dev_vp, nPixels*sizeof(float));
	err = cudaMalloc((void**)&dev_ktrans, nPixels*sizeof(float));
	err = cudaMalloc((void**)&dev_kep, nPixels*sizeof(float));
	err = cudaMalloc((void**)&dev_converged, nPixels*sizeof(float));
	err = cudaMalloc((void**)&dev_iter, nPixels*sizeof(float));
	err = cudaMalloc((void**)&dev_res, nPixels*sizeof(float));
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
	printf("|\\\n");
	printf("|* Device total memory: %lluMB\n|* Device free memory: %lluMB\n",(total/1024)/1024,(free/1024)/1024);
 	printf("|/\n");

	/*********************************/
	/* Copy data from host to device */
	/*********************************/
	printf("* Copying data to device...\n");
	err = cudaMemcpy(dev_time,time,nt*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_AIF,Cp_re,nt*sizeof(float),cudaMemcpyHostToDevice);
	err = cudaMemcpy(dev_mriSizes,mriSizes,(nsd+1)*sizeof(int),cudaMemcpyHostToDevice);
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
  printf("* Kernel execution...\n");  
	dim3 block(8,8);
	dim3 grid(mriSizes[0]/block.x,mriSizes[1]/block.y,1);
	if(cudaThreadSetLimit(cudaLimitStackSize,8*stackSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	} 
	if(cudaThreadSetLimit(cudaLimitMallocHeapSize,4*heapSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
	if(cudaThreadGetLimit(&stackSize,cudaLimitStackSize)){
  	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	}
 	printf("|\\\n");
	printf("|* Cuda Thread Limit Stack Size = %lluKB\n",stackSize/1024);
	if(cudaThreadGetLimit(&heapSize,cudaLimitMallocHeapSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
 	printf("|* Cuda Thread Limit Malloc Heap Size = %lluMB\n",(heapSize/1024)/1024);
	for(int i=0;i<mriSizes[2]/grid.z;i++){
		int pOffset = nPixels*i;
		err = cudaMemcpy(dev_Ct,Ct.getDataPointer()+pOffset*nt,nPixels*nt*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_vp,vp.getDataPointer()+pOffset,nPixels*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_ktrans,ktrans.getDataPointer()+pOffset,nPixels*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_kep,kep.getDataPointer()+pOffset,nPixels*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_converged,converged.getDataPointer()+pOffset,nPixels*sizeof(float),cudaMemcpyHostToDevice);
		NLLSQkernel<<<grid,block>>>(nsd,nt,dev_mriSizes,difft,dev_Ct,dev_time,dev_AIF,
																dev_Cp,dev_fftCp,dev_vp,dev_ktrans,dev_kep,dev_converged,
																dev_iter,dev_res,i);
		err = cudaMemcpy(vp.getDataPointer()+pOffset,dev_vp,nPixels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(ktrans.getDataPointer()+pOffset,dev_ktrans,nPixels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(kep.getDataPointer()+pOffset,dev_kep,nPixels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(converged.getDataPointer()+pOffset,dev_converged,nPixels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(iter.getDataPointer()+pOffset,dev_iter,nPixels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(res.getDataPointer()+pOffset,dev_res,nPixels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(CtN.getDataPointer()+pOffset*nt,dev_Ct,nPixels*nt*sizeof(float),cudaMemcpyDeviceToHost);
		if( (err=cudaDeviceSynchronize()) != cudaSuccess){
			printf("err = %d\n",err);
 			fprintf(stderr, "Cuda error: Failed to synchronize\n");
 			return 0;
		}
		err = cudaEventRecord(stop, 0);
		err = cudaEventSynchronize(stop);
		err = cudaEventElapsedTime(&elapsed, start, stop);
		printf("|* [NLLSQkernel  %d] The elapsed time in gpu was %.2f s.\n",i+1,elapsed/1000);
	}
	if(cudaMemGetInfo(&free,&total) != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to get device memory info\n");
    return 0;
  }
	printf("|* Device total memory: %lluMB\n|* Device free memory: %lluMB\n",(total/1024)/1024,(free/1024)/1024);
	err = cudaEventRecord(stop, 0);
	err = cudaEventSynchronize(stop);
	err = cudaEventElapsedTime(&elapsed, start, stop);
	err = cudaEventDestroy(start);
	err = cudaEventDestroy(stop);
	printf("|* The elapsed time in gpu was %.2f s.\n", elapsed/1000);
 	printf("|/\n");

	/*********************************/
	/* Copy data from device to host */
	/*********************************/
	printf("* Copying data to host...\n");
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
        ve.setDataValue(aux_offset,ktrans.getDataValue(aux_offset)/kep.getDataValue(aux_offset));
      }
    }
  }
	CtN.setData(Ct_aux);	/* Copy the CtN */

	/******************/
	/* Save solutions */
	/******************/
	printf("* Saving solution...\n");
 	printf("|\\\n");
	vp.writeToBinary("vp.dat");
  ktrans.writeToBinary("ktrans.dat");
  kep.writeToBinary("kep.dat");
	ve.writeToBinary("ve.dat");
  converged.writeToBinary("converged.dat");
  iter.writeToBinary("iterations.dat");
  res.writeToBinary("residuals.dat");
  CtN.writeToBinary("CtN.dat");
	printf("|* Global residual = %f\n",global_res);
	printf("|* %.3f%% voxels converged\n",(float)voxelsConverged/nVoxels*100);
 	printf("|/\n");

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

	err = cudaEventRecord(mainStop, 0);
	err = cudaEventSynchronize(mainStop);
	err = cudaEventElapsedTime(&mainTime, mainStart, mainStop);
	printf("* Total elapsed time was %.2f s.\n",mainTime/1000);
  printf("* Finished!\n");
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
//			if(i!=0){		/* Arreglo casero cuando kep muy grande */
//				for(int k=0;k<NPARAMS-1;k++){
//					m[k][i]=0;			
//				}
//				m[i][i]=1;
//			}
//			else{
      	printf("Pivot ~ 0 en solver3x3().");
      	assert(0);
//			}
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
	//int pz = blockIdx.z;
	int pz = zlayer*gridDim.z+blockIdx.z;
	if(px>=mriSizes[0]||py>=mriSizes[1]||pz>=mriSizes[2]) return;

  float *J = new float[NPARAMS*nt];					/* Jacobian for the NLLSQ method */
  float *Ct_i = new float[nt];  						/* Ct data from 1 voxel */
  float *Ct_n = new float[nt];							/* Ct numerical approximation */
  float *dCt = new float[nt];								/* Diference between Ct_i and Ct_n */
  complex *Exp = new complex[NX];						/* Exponential function */
  complex *fftExp = new complex[NX];				/* fft(Exp) or fft(Exp.*t) */
  complex *convol = new complex[NX];				/* ifft(fftCp.*fftExp) */
  float *integral = new float[nt];					/* Truncated real part of convol using fft(Exp) */
  float *dintegral = new float[nt];					/* Truncated real part of convol using fft(Exp.*t) */

	/*********************************************/
	/* ¿Conviene copiar los datos a cada thread? */
	/*********************************************/
	float *Cp_re = new float[nt];
	float *time = new float[nt];
	complex *Cp = new complex[NX];
	complex *fftCp = new complex[NX];
	for(int i=0;i<nt;i++){
		Cp_re[i] = AIF[i];
		time[i] = t[i];
	}
	for(int i=0;i<NX;i++){
		Exp[i] = 0.0;
		Cp[i] = Cp_g[i];	
		fftCp[i] = fftCp_g[i];	
	}
/*
	if(px==0&&py==0) {
		for(int l=0;l<nt;l++) printf("%f %f\n",Cp_re[l],time[l]);
		for(int l=0;l<NX;l++) printf("%f %f %f %f\n",Exp[l].re(),Cp[l].re(),fftCp[l].re(),fftCp[l].im());
	}
*/
	/*********************************************/
	/*********************************************/
//  for(int py=0;py<mriSizes[1];py++){
//    for(int pz=0;pz<mriSizes[2];pz++){
	{{
	    /**********************/
      /* Initial parameters */
	    /**********************/
      int param_offset = py*mriSizes[0] + px;
      int data_offset = param_offset*nt;
      float vp0=0.5,ktrans0=0.1/60,kep0=0.1/60;
      for(int l=0;l<nt;l++){
		    Ct_i[l] = Ct[data_offset + l];								/* Ct data of voxel(px,py,pz) */
      	Exp[l] = exp(-kep0*time[l]);									/* Exponential function */
	    }
      CFFT::Forward(Exp,fftExp,NX);										/* fft(Exp) */
      for(int l=0;l<NX;l++)
        fftExp[l] *= fftCp[l];												/* fftExp.*fftCp */
      CFFT::Inverse(fftExp,convol,NX);								/* ifft(fftCp.*fftExp) */
      for(int l=0;l<nt;l++){
        integral[l] = convol[l].re()*difft;						/* Truncated real part of convol using fft(Exp) */
        Ct_n[l] = vp0*Cp_re[l] + ktrans0*integral[l];	/* Ct numerical approximation */
        dCt[l] = Ct_i[l] - Ct_n[l];										/* Diference between Ct_i and Ct_n */
      }
      float r_old = SSQ(dCt,nt);
      
  /*		printf("%d %d %d %d %d\n",nt,NX,nsd,param_offset,data_offset);
		  printf("%f %f %f %f\n",difft,vp0,ktrans0,kep0);
		  if(px==0&&py==0&&pz==0) {
			  for(int l=0;l<nt;l++) printf("%f %f %f %f %f\n",Ct_i[l],Exp[l].re(),integral[l],Ct_n[l],dCt[l]);
			  for(int l=0;l<NX;l++) printf("%f %f %f %f\n",Exp[l].re(),fftExp[l].re(),fftExp[l].im(),convol[l].re());
		  return;
		  }
  */
	    /***********************************/
      /* Non-Liniear Least Square Method */
	    /***********************************/
      int its = 0;
      float dB[NPARAMS]={1e10,1e10,1e10};
	    float beta[NPARAMS]={vp0,ktrans0,kep0};  		/* SHOULD CHANGE IF NPARAMS CHANGE */
	    float beta_min[NPARAMS]={vp0,ktrans0,kep0};	/* SHOULD CHANGE IF NPARAMS CHANGE */
      float dr,dr_rel=1e10,dB_rel=1;
      float r_min=1e10,r_new;
      while( its<MAXIT && (norm(dB,NPARAMS)>1e-3||(dr_rel)>1e-3||r_old==0.0) && kep0>1e-9 ){        
     		its++;
        for(int l=0;l<nt;l++)
          Exp[l] *= time[l];
        CFFT::Forward(Exp,fftExp,NX);
        for(int l=0;l<NX;l++)
          fftExp[l] *= fftCp[l];
        CFFT::Inverse(fftExp,convol,NX);
        for(int l=0;l<nt;l++){
          dintegral[l] = convol[l].re()*difft;
          J[l] = Cp_re[l];    
          J[nt+l] = integral[l];
          J[2*nt+l] = -ktrans0*dintegral[l];
        }
  /*
		  if(px==0&&py==0&&pz==0&&its==2) {
			  for(int l=0;l<nt;l++) printf("%f %f %f %f %f\n",dintegral[l],J[l],J[nt+l],J[2*nt+l],dCt[l]);
			  for(int l=0;l<NX;l++) printf("%f %f %f\n",fftCp[l].re(),fftCp[l].im(),convol[l].re());
			  for(int l=0;l<NX;l++) printf("%f %f %f %f\n",Exp[l].re(),Exp[l].im(),fftExp[l].re(),fftExp[l].im());
		  }
  */
        solveNLLSQ(J,dCt,dB,nt);
  /*
		  if(px==0&&py==0&&pz==0&&its==2) 
			  printf("%f,%f,%f\n",dB[0],dB[1],dB[2]);
  */
				dB_rel = 0;
		    for(int i=0;i<NPARAMS;i++){
          if((beta[i]+dB[i])<0)
            dB[i] = -3.0/4.0*beta[i];      
            //dB[i] = -beta[i];      
			    //dB[i] /= 2;
//			    beta[i] += dB[i];
//					dB_rel += fabs(dB[i]/beta[i]);
		    } 
//       	vp0 = beta[0];
//				if(beta[1]<1e-6) beta[2]=1e-6;
//				if(beta[2]>1) beta[2]=0.1;
//       	ktrans0 = beta[1];
//       	kep0 = beta[2];
				if(isnan(beta[0])||isnan(beta[1])||isnan(beta[2])){
					beta_min[0] = 0;
					beta_min[1] = 0;
					beta_min[2] = 0;
					r_min = SSQ(Ct_i,nt);
          break;
        }
/**/
				int flag = 1,sub_it=0;
				while(flag && sub_it<5){
					sub_it++;
       		vp0 = beta[0]+dB[0];
       		ktrans0 = beta[1]+dB[1];
       		kep0 = beta[2]+dB[2];
					if(ktrans0<1e-6) kep0=1e-6;
					if(kep0>1) kep0=0.1/60;
/**/
       		for(int l=0;l<nt;l++)
          	Exp[l] = exp(-kep0*time[l]);
        	CFFT::Forward(Exp,fftExp,NX);
        	for(int l=0;l<NX;l++)
          	fftExp[l] *= fftCp[l];
        	CFFT::Inverse(fftExp,convol,NX);
        	for(int l=0;l<nt;l++){
          	integral[l] = convol[l].re()*difft;
          	Ct_n[l] = vp0*Cp_re[l] + ktrans0*integral[l];
          	dCt[l] = Ct_i[l] - Ct_n[l];
        	}
        	r_new = SSQ(dCt,nt);
/**/
					if(r_new>r_old){
						for(int i=0;i<NPARAMS;i++)
							dB[i] /= 2;
					}
					else
						flag = 0;
				}
				dB_rel = 0;
		    for(int i=0;i<NPARAMS;i++){
					beta[i] += dB[i];
					dB_rel += fabs(dB[i]/beta[i]);
		    } 
/**/
  /*
		  if(px==0&&py==0&&pz==0&&its==2) {
			  for(int l=0;l<nt;l++) printf("%f %f %f %f\n",integral[l],Ct_i[l],Ct_n[l],dCt[l]);
			  for(int l=0;l<NX;l++) printf("%f\n",convol[l].re());
			  for(int l=0;l<NX;l++) printf("%f %f %f %f\n",Exp[l].re(),Exp[l].im(),fftExp[l].re(),fftExp[l].im());
		  }
  */
        if(r_new<r_min){
          r_min=r_new;
			    for(int i=0;i<NPARAMS;i++)
          	beta_min[i] = beta[i];
        }  
        dr_rel = dr = fabs(r_old-r_new);
				if(r_old>1e-6)
					dr_rel /= r_old;
/* 
	if(px==313&&py==219&&pz==23){ 
	 	printf("it: %d %f %f %f\n",its,dB[0],dB[1],dB[2]);
	 	printf("it: %d %f %f %f\n",its,beta[0],beta[1],beta[2]);
	 	printf("it: %d %f %f %f %f\n",its,r_old,r_new,dr,dr_rel);
  }
*/
        r_old = r_new;
      }
      if(its!=MAXIT)	
		    converged[param_offset] = 1;
      vp[param_offset] = beta_min[0];
      ktrans[param_offset] = beta_min[1];
//			if(beta_min[1]<1e-9) beta_min[2]=0;
      kep[param_offset] = beta_min[2];
      iter[param_offset] = its;
      res[param_offset] = r_min;
			for(int l=0;l<nt;l++)
				Ct[data_offset+l] = Ct_n[l];
  /*
		  if(px==0&&py==0&&pz==0) {
			  printf("its: %d\n",its);
			  return;
		  }
  */
		if(param_offset>=mriSizes[0]*mriSizes[1]){
			printf("param_offset = %d, max = %d",param_offset,mriSizes[0]*mriSizes[1]);
			assert(0);
		}
//	 	printf("Voxels(%d,%d,%d)... its: %d\n",px+1,py+1,pz+1,its);
//	 	if(threadIdx.x==0&&threadIdx.y==0) printf("Block (%d,%d,%d)\n",blockIdx.x,blockIdx.y,pz);//blockIdx.z);
    }
	}
	
	delete[] J;
  delete[] Ct_i;
  delete[] Ct_n;
  delete[] dCt;
  delete[] Exp;
  delete[] fftExp;
  delete[] convol;
  delete[] integral;
  delete[] dintegral;
	delete[] Cp_re;
	delete[] time;
	delete[] Cp;
	delete[] fftCp;

	return;
}
