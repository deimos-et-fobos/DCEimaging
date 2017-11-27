#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

int findkey(FILE *fid, const char *str){
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

void cfgreading(const char *cfgfile, dim3 *threadsPerBlock, dim3 *blocksPerGrid, float *stackSizeFactor, float *heapSizeFactor){
	int def;
	char line[80];
	dim3 TpB(1,1,1);
	dim3 BpG(1,1,1);
	FILE *cfg=fopen(cfgfile,"r");
	printf("* Reading 'DCEimaging.cfg'...\n");
	if(cfg==NULL){
		printf("|\\\n");
		printf("|* Can't open '%s'\n",cfgfile);
		printf("|* Using default threads per block: (%d,%d,%d)\n",threadsPerBlock->x,threadsPerBlock->y,threadsPerBlock->z);
		printf("|* Using default blocks per grid: (%d,%d,%d)\n",blocksPerGrid->x,blocksPerGrid->y,blocksPerGrid->z);
		printf("|/\n");
		return;
	}
	printf("|\\\n");

	/* THREADS_PER_BLOCK */
	def=1;
	if(findkey(cfg,"THREADS_PER_BLOCK")){
		if(fgets(line,80,cfg)!=NULL){
			if(sscanf(line,"%d %d %d",&TpB.x,&TpB.y,&TpB.z)>0){
				threadsPerBlock->x = TpB.x;			
				threadsPerBlock->y = TpB.y;			
				threadsPerBlock->z = TpB.z;			
				def=0;
			}
		}
	}
	if(def)
		printf("|* Using default threads per block: (%d,%d,%d)\n",threadsPerBlock->x,threadsPerBlock->y,threadsPerBlock->z);
	else
		printf("|* Threads per block: (%d,%d,%d)\n",threadsPerBlock->x,threadsPerBlock->y,threadsPerBlock->z);

	/* BLOCKS_PER_GRID */
	def=1;
	if(findkey(cfg,"BLOCKS_PER_GRID")){
		if(fgets(line,80,cfg)!=NULL){
			if(sscanf(line,"%d %d %d",&BpG.x,&BpG.y,&BpG.z)>0){
				blocksPerGrid->x = BpG.x;			
				blocksPerGrid->y = BpG.y;			
				blocksPerGrid->z = BpG.z;			
				def=0;
			}
		}
	}
	if(def)
		printf("|* Using default blocks per grid: (%d,%d,%d)\n",blocksPerGrid->x,blocksPerGrid->y,blocksPerGrid->z);
	else
		printf("|* Blocks per grid: (%d,%d,%d)\n",blocksPerGrid->x,blocksPerGrid->y,blocksPerGrid->z);

	/* STACK_SIZE_FACTOR */
	def=1;
	if(findkey(cfg,"STACK_SIZE_FACTOR")){
		if(fgets(line,80,cfg)!=NULL){
			if(sscanf(line,"%f",stackSizeFactor)>0)
				def=0;
		}
	}
	if(def)
		printf("|* Using default Cuda Thread Limit Stack Size factor: %.1f\n",*stackSizeFactor);
	else
		printf("|* Cuda Thread Limit Stack Size factor: %.1f\n",*stackSizeFactor);

	/* HEAP_SIZE_FACTOR */
	def=1;
	if(findkey(cfg,"HEAP_SIZE_FACTOR")){
		if(fgets(line,80,cfg)!=NULL){
			if(sscanf(line,"%f",heapSizeFactor)>0)
				def=0;
		}
	}
	if(def)
		printf("|* Using default Cuda Thread Limit Malloc Heap Size factor: %.1f\n",*heapSizeFactor);
	else
		printf("|* Cuda Thread Limit Malloc Heap Size factor: %.1f\n",*heapSizeFactor);

	printf("|/\n");
	fclose(cfg);
}
