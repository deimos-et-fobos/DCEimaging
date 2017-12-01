class cfgStr{
	private:
		int NLLSQMaxIt;
		int fftLength;
		float stackSizeFactor;
		float heapSizeFactor;
		dim3 threadsPerBlock;
		dim3 blocksPerGrid;
		FILE *fid;
	public:
		cfgStr(){
			NLLSQMaxIt = NLLSQMAXIT;
			fftLength = FFTLENGTH;
			stackSizeFactor = STACKSIZEFACTOR;
			heapSizeFactor = HEAPSIZEFACTOR;
			threadsPerBlock.x = THREADSPERBLOCK_X;
			threadsPerBlock.y = THREADSPERBLOCK_Y;
			threadsPerBlock.z = THREADSPERBLOCK_Z;
			blocksPerGrid.x = BLOCKSPERGRID_X;
			blocksPerGrid.y = BLOCKSPERGRID_Y;
			blocksPerGrid.z = BLOCKSPERGRID_Z;
			fid = NULL;
		};
    int findKey(const char *str);
    void cfgRead(const char *cfgfile);
    int checkCfgErrors();
	  int sizesError(int sz);
    void setFFTLength(int n);
		int getNLLSQMaxIt(){return NLLSQMaxIt;};
		int getFFTLength(){return fftLength;};
		float getStackSizeFactor(){return stackSizeFactor;};
		float getHeapSizeFactor(){return heapSizeFactor;};
		dim3 getThreadsPerBlock(){return threadsPerBlock;};
		dim3 getBlocksPerGrid(){return blocksPerGrid;};
		~cfgStr(){};
};

int cfgStr::findKey(const char *str){
	bool rew = false;
  char line[80];  
  int str_len = strlen(str);
	while(1){
		if(fgets(line,80,fid)!=NULL){
			if(line[0]!='*') continue;
			if(strncmp(&line[1],str,str_len)==0){
				if(line[str_len+1]==' ' || line[str_len+1]=='\0' || line[str_len+1]=='\n')
					return 1;
			}
		}
		else{
			if(rew) break;
			rew = true;
			rewind(fid);	
		}
	}
	return 0;
}

void cfgStr::cfgRead(const char *cfgfile){
	int def;
	char line[80];
	dim3 TpB(1,1,1);
	dim3 BpG(1,1,1);
	fid=fopen(cfgfile,"r");
	printf("* Reading 'DCEimaging.cfg'...\n");
	if(fid==NULL){
		printf("|\\\n");
		printf("|* Can't open '%s'\n",cfgfile);
		printf("|* Using default threads per block: (%d,%d,%d)\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
		printf("|* Using default blocks per grid: (%d,%d,%d)\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);
		printf("|* Using default Cuda Thread Limit Stack Size factor: %.1f\n",stackSizeFactor);
		printf("|* Using default Cuda Thread Limit Malloc Heap Size factor: %.1f\n",heapSizeFactor);
		printf("|* Using default NLLSQ Method's Maximum Iterations: %d\n",NLLSQMaxIt);
		printf("|* Using default FFT vector length: %d\n",fftLength);
		/* check if n is a power of 2 */
		int n=1;
		while(n<fftLength) n*=2;
		if(n>fftLength){
			fftLength = n;
			printf("|* FFT vector lenght must be power of 2. Changing to the next power of 2: %d\n",fftLength);
		}
		printf("|/\n");
		return;
	}
	printf("|\\\n");

	/* THREADS_PER_BLOCK */
	def=1;
	if(findKey("THREADS_PER_BLOCK")){
		if(fgets(line,80,fid)!=NULL){
			if(sscanf(line,"%d %d %d",&TpB.x,&TpB.y,&TpB.z)>0){
				threadsPerBlock.x = TpB.x;			
				threadsPerBlock.y = TpB.y;			
				threadsPerBlock.z = TpB.z;			
				def=0;
			}
		}
	}
	if(def)
		printf("|* Using default threads per block: (%d,%d,%d)\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);
	else
		printf("|* Threads per block: (%d,%d,%d)\n",threadsPerBlock.x,threadsPerBlock.y,threadsPerBlock.z);

	/* BLOCKS_PER_GRID */
	def=1;
	if(findKey("BLOCKS_PER_GRID")){
		if(fgets(line,80,fid)!=NULL){
			if(sscanf(line,"%d %d %d",&BpG.x,&BpG.y,&BpG.z)>0){
				blocksPerGrid.x = BpG.x;			
				blocksPerGrid.y = BpG.y;			
				blocksPerGrid.z = BpG.z;			
				def=0;
			}
		}
	}
	if(def)
		printf("|* Using default blocks per grid: (%d,%d,%d)\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);
	else
		printf("|* Blocks per grid: (%d,%d,%d)\n",blocksPerGrid.x,blocksPerGrid.y,blocksPerGrid.z);

	/* STACK_SIZE_FACTOR */
	def=1;
	if(findKey("STACK_SIZE_FACTOR")){
		if(fgets(line,80,fid)!=NULL){
			if(sscanf(line,"%f",&stackSizeFactor)>0)
				def=0;
		}
	}
	if(def)
		printf("|* Using default Cuda Thread Limit Stack Size factor: %.1f\n",stackSizeFactor);
	else
		printf("|* Cuda Thread Limit Stack Size factor: %.1f\n",stackSizeFactor);

	/* HEAP_SIZE_FACTOR */
	def=1;
	if(findKey("HEAP_SIZE_FACTOR")){
		if(fgets(line,80,fid)!=NULL){
			if(sscanf(line,"%f",&heapSizeFactor)>0)
				def=0;
		}
	}
	if(def)
		printf("|* Using default Cuda Thread Limit Malloc Heap Size factor: %.1f\n",heapSizeFactor);
	else
		printf("|* Cuda Thread Limit Malloc Heap Size factor: %.1f\n",heapSizeFactor);

	/* FFT_LENGTH */
	def=1;
	if(findKey("FFT_LENGTH")){
		if(fgets(line,80,fid)!=NULL){
			if(sscanf(line,"%d",&fftLength)>0)
				def=0;
		}
	}
	/* check if n is a power of 2 */
	int n=1;
	while(n<fftLength) n*=2;
	if(n>fftLength){
		printf("|* FFT vector lenght (%d) must be power of 2. Changing to the next power of 2\n",fftLength);
		fftLength = n;
		def=0;
	}
	if(def)
		printf("|* Using default FFT vector length: %d\n",fftLength);
	else
		printf("|* FFT vector length: %d\n",fftLength);

	/* NLLSQ_MAXIT */
	def=1;
	if(findKey("NLLSQ_MAXIT")){
		if(fgets(line,80,fid)!=NULL){
			if(sscanf(line,"%d",&NLLSQMaxIt)>0)
				def=0;
		}
	}
	if(def)
		printf("|* Using default NLLSQ Method's Maximum Iterations: %d\n",NLLSQMaxIt);
	else
		printf("|* Maximum Iterations: %d\n",NLLSQMaxIt);

	printf("|/\n");
	fclose(fid);
}

int cfgStr::checkCfgErrors(){
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
	return flag;
}

int cfgStr::sizesError(int sz){
	if(sz%(threadsPerBlock.z*blocksPerGrid.z)){
		printf("MRI z-dimension (%d) must my multiple of threadsPerBlock.z*blocksPerGrid.z (%d)\n",sz,(threadsPerBlock.z*blocksPerGrid.z));	
		return 1;
	}
	return 0;
}

void cfgStr::setFFTLength(int n){
	if(n>fftLength){
		printf("|\\\n");
		printf("|* MRI time samples > FFT vector length (%d > %d)\n",n,fftLength);	
		fftLength=1;
		while(fftLength<n) fftLength*=2;
		printf("|* Changing the FFT vector length the next power of 2\n");
		printf("|* FFT vector length: %d\n",fftLength);
		printf("|/\n");
	}
}
