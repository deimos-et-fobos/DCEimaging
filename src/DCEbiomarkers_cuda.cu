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
#include "cfgreading.h"

#define NX 64
#define MAXIT 200
#define NPARAMS 3

/* Default values */
#define THREADSPERBLOCK_X 8
#define THREADSPERBLOCK_Y 8
#define THREADSPERBLOCK_Z 1
#define BLOCKSPERGRID_X 64
#define BLOCKSPERGRID_Y 64
#define BLOCKSPERGRID_Z 1
#define HEAPSIZEFACTOR 8
#define STACKSIZEFACTOR 8

template <typename T> void writeToBinary(const char *filename, T *data, int ndata, int *sizes, int dim);
template <typename T> void readFromBinary(const char *filename, T **data, int *ndata, int **sizes, int *dim);
template <typename T> __host__ __device__ T norm(T *vec, int n);
template <typename T> __host__ __device__ T SSQ(T *vec, int n);
__device__ void solveNLLSQ(float *J, float *dCt, float *dB, int npoints);
__device__ void gaussSolver(float *A,float *b, float *x);
__global__ void NLLSQkernel(int nsd, int nt, int *mriSizes, float difft, float *Ct, float *t, float *AIF, 
								complex *Cp_g, complex *fftCp_g, float *vp, float *ktrans, float *kep, float *converged,
								float *iter, float *res, dim3 gridIdx);
__host__ int checkInitErrors(dim3 threadsPerBlock, dim3 blocksPerGrid, int mri_zdim, int nt);

int main(void){
	float mainTime=0;
  cudaError_t err = cudaSuccess;				// Error code to check return values for CUDA calls
	size_t free,total,heapSize,stackSize;
	dim3 threadsPerBlock(THREADSPERBLOCK_X,THREADSPERBLOCK_Y,THREADSPERBLOCK_Z);
	dim3 blocksPerGrid(BLOCKSPERGRID_X,BLOCKSPERGRID_Y,BLOCKSPERGRID_Z);
	float stackSizeFactor = STACKSIZEFACTOR;
	float heapSizeFactor = HEAPSIZEFACTOR;
	char cfgfile[]="DCEimaging.cfg";
	cudaEvent_t mainStart, mainStop;
	err = cudaEventCreate(&mainStart);
	err = cudaEventCreate(&mainStop);
	err = cudaEventRecord(mainStart, 0);

	/* Reading configuration file */
  cfgreading(cfgfile,&threadsPerBlock,&blocksPerGrid,&stackSizeFactor,&heapSizeFactor);

	/* Check errors in the GPU initilized variables */
	if(checkInitErrors(threadsPerBlock,blocksPerGrid,0,0)) return 0;

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
  float *Ct_aux = new float[nVoxels*nt];/* Here will be reordered C_t */
  float difft = (time[nt-1]-time[0])/(nt-1);	/* Average temporary step */

	/**************************************************/
	/* Number of threads, blocks and grids per Kernel */	
	/**************************************************/
	/* Check that mriSizes[2] is multiple of (threadsPerBlock.z*blocksPerGrid.z) */
	if(checkInitErrors(threadsPerBlock,blocksPerGrid,mriSizes[2],nt)) return 0;
	dim3 gridsPerKernel(1+(mriSizes[0]-1)/(threadsPerBlock.x*blocksPerGrid.x),1+(mriSizes[1]-1)/(threadsPerBlock.y*blocksPerGrid.y),mriSizes[2]/(threadsPerBlock.z*blocksPerGrid.z));
	int sliceVoxels = nVoxels/gridsPerKernel.z; 	// Number of voxels per MRI slice

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
	err = cudaMalloc((void**)&dev_Ct, sliceVoxels*nt*sizeof(float));
	err = cudaMalloc((void**)&dev_vp, sliceVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_ktrans, sliceVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_kep, sliceVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_converged, sliceVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_iter, sliceVoxels*sizeof(float));
	err = cudaMalloc((void**)&dev_res, sliceVoxels*sizeof(float));
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
	dim3 gridIdx(0,0,0);
 	printf("|\\\n");
	printf("|* Threads per block: (%d,%d,%d)\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	printf("|* Blocks per grid: (%d,%d,%d)\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);
	printf("|* Grids per kernel: (%d,%d,%d)\n",gridsPerKernel.x,gridsPerKernel.y,gridsPerKernel.z);
	if(cudaThreadSetLimit(cudaLimitStackSize,stackSizeFactor*stackSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Stack Size\n");
   	return 0;
	} 
	if(cudaThreadSetLimit(cudaLimitMallocHeapSize,heapSizeFactor*heapSize)){
   	fprintf(stderr, "Cuda error: Failed to get Thread Limit Malloc Heap Size\n");
   	return 0;
	}
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
	for(int k=0;k<gridsPerKernel.z;k++){
		gridIdx.z = k;
		int pOffset = sliceVoxels*k;
		err = cudaMemcpy(dev_Ct,Ct.getDataPointer()+pOffset*nt,sliceVoxels*nt*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_vp,vp.getDataPointer()+pOffset,sliceVoxels*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_ktrans,ktrans.getDataPointer()+pOffset,sliceVoxels*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_kep,kep.getDataPointer()+pOffset,sliceVoxels*sizeof(float),cudaMemcpyHostToDevice);
		err = cudaMemcpy(dev_converged,converged.getDataPointer()+pOffset,sliceVoxels*sizeof(float),cudaMemcpyHostToDevice);
		for(int i=0;i<gridsPerKernel.x;i++){
			gridIdx.x = i;
			for(int j=0;j<gridsPerKernel.y;j++){
				gridIdx.y = j;
				NLLSQkernel<<<blocksPerGrid,threadsPerBlock>>>(nsd,nt,dev_mriSizes,difft,dev_Ct,dev_time,dev_AIF,
										dev_Cp,dev_fftCp,dev_vp,dev_ktrans,dev_kep,dev_converged,dev_iter,dev_res,gridIdx);
			}
			if( (err=cudaDeviceSynchronize()) != cudaSuccess){
				printf("err = %d\n",err);
 				fprintf(stderr, "Cuda error: Failed to synchronize\n");
 				return 0;
			}
		}
		err = cudaMemcpy(vp.getDataPointer()+pOffset,dev_vp,sliceVoxels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(ktrans.getDataPointer()+pOffset,dev_ktrans,sliceVoxels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(kep.getDataPointer()+pOffset,dev_kep,sliceVoxels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(converged.getDataPointer()+pOffset,dev_converged,sliceVoxels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(iter.getDataPointer()+pOffset,dev_iter,sliceVoxels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(res.getDataPointer()+pOffset,dev_res,sliceVoxels*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaMemcpy(CtN.getDataPointer()+pOffset*nt,dev_Ct,sliceVoxels*nt*sizeof(float),cudaMemcpyDeviceToHost);
		err = cudaEventRecord(stop, 0);
		err = cudaEventSynchronize(stop);
		err = cudaEventElapsedTime(&elapsed, start, stop);
		printf("|* [NLLSQkernel (:,:,%2d)] The elapsed time in gpu was %.2f s\n",k+1,elapsed/1000);
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
	cudaFree(dev_iter);
	cudaFree(dev_res);
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
								float *iter, float *res, dim3 gridIdx){

	int px = threadIdx.x + blockDim.x*blockIdx.x + gridIdx.x*blockDim.x*gridDim.x;
	int py = threadIdx.y + blockDim.y*blockIdx.y + gridIdx.y*blockDim.y*gridDim.y;
	int pz = threadIdx.z + blockDim.z*blockIdx.z;
	//int pz = threadIdx.z + blockDim.z*blockIdx.z + gridIdx.z*blockDim.z*gridDim.z;
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
      int param_offset = pz*mriSizes[0]*mriSizes[1] + py*mriSizes[0] + px;
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
      float dr_rel=1e10,dB_rel=1;
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
        dr_rel = fabs(r_old-r_new);
				if(r_old>1e-6)
					dr_rel /= r_old;
/* 
	if(px==313&&py==219&&pz==23){ 
	 	printf("it: %d %f %f %f\n",its,dB[0],dB[1],dB[2]);
	 	printf("it: %d %f %f %f\n",its,beta[0],beta[1],beta[2]);
	 	printf("it: %d %f %f %f %f\n",its,r_old,r_new,dr_rel);
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
		if(param_offset>=mriSizes[0]*mriSizes[1]*blockDim.z*gridDim.z){
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

int checkInitErrors(dim3 threadsPerBlock, dim3 blocksPerGrid, int mri_zdim, int nt){
	int flag = 0;
	cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if((threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z)%deviceProp.warpSize){
    printf("threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z (%d x %d x %d = %d) must be divisible by the Warp Size (%d).\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z,threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z,deviceProp.warpSize);
    flag = 1;
  }
  if((threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z)>deviceProp.maxThreadsPerBlock){
    printf("threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z (%d x %d x %d = %d) must be <= %d.\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z,threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z,deviceProp.maxThreadsPerBlock);
    flag = 1;
  }
  if(threadsPerBlock.x>deviceProp.maxThreadsDim[0]){
    printf("threadsPerBlock.x (%d) must be <= %d.\n",threadsPerBlock.x,deviceProp.maxThreadsDim[0]);
    flag = 1;
  }
  if(threadsPerBlock.y>deviceProp.maxThreadsDim[1]){
    printf("threadsPerBlock.y (%d) must be <= %d.\n",threadsPerBlock.y,deviceProp.maxThreadsDim[1]);
    flag = 1;
  }
  if(threadsPerBlock.z>deviceProp.maxThreadsDim[2]){
    printf("threadsPerBlock.z (%d) must be <= %d.\n",threadsPerBlock.z,deviceProp.maxThreadsDim[2]);
    flag = 1;
  }
  if(blocksPerGrid.x>deviceProp.maxGridSize[0]){
    printf("blocksPerGrid.x (%d) must be <= %d.\n",blocksPerGrid.x,deviceProp.maxGridSize[0]);
    flag = 1;
  }
  if(blocksPerGrid.y>deviceProp.maxGridSize[1]){
    printf("blocksPerGrid.y (%d) must be <= %d.\n",blocksPerGrid.y,deviceProp.maxGridSize[1]);
    flag = 1;
  }
  if(blocksPerGrid.z>deviceProp.maxGridSize[2]){
    printf("blocksPerGrid.z (%d) must be <= %d.\n",blocksPerGrid.z,deviceProp.maxGridSize[2]);
    flag = 1;
  }
	if(mri_zdim%(threadsPerBlock.z*blocksPerGrid.z)){
		printf("MRI z-dimension (%d) must my multiple of threadsPerBlock.z*blocksPerGrid.z (%d)\n",mri_zdim,(threadsPerBlock.z*blocksPerGrid.z));	
		flag = 1;
	}
	if(nt>NX){
		printf("MRI time samples > fft vector lenght paramater (%d > %d)\n",nt,NX);	
		flag = 1;
	}
	return flag;
}
