/****************/
/* Class MATRIX */
/****************/

template <typename T>
class matrix{
		int dim;
		int *sz;
		int nd;
		T 	*data;

	public:
		/* Constructors */
		matrix(): dim(0),sz(NULL),nd(0),data(NULL){};
		matrix(const char *filename){ this->readFromBinary(filename); };
		matrix(int ndim, int *sizes, int ndata): dim(ndim),nd(ndata){
			sz = new int[ndim];
			assert(sz);
			for(int i=0;i<ndim;i++)
				sz[i] = sizes[i];
			data = new T[ndata];
			assert(data);
			for(int i=0;i<ndata;i++)
					data[i] = 0;
		};
		matrix(int ndim, int *sizes, int ndata, T *d): dim(ndim),nd(ndata){
			sz = new int[ndim];
			assert(sz);
			for(int i=0;i<ndim;i++)
				sz[i] = sizes[i];
			data = new T[ndata];
			assert(data);
			if(d==NULL){
				for(int i=0;i<ndata;i++)
					data[i] = 0;
			}else{
				for(int i=0;i<ndata;i++)
					data[i] = d[i];
			}
		};

		/* Destructor */
		~matrix(){ this->free(); };

		/* Mutators */
		void setData(T *d){	
			if(data==NULL)	data = new T[nd];
			assert(data);
			for(int i=0;i<nd;i++)
				data[i]=d[i];
		};
		inline void setDataValue(int idx, T val){	
			if(idx<nd) 	data[idx]=val; 
			else				assert(idx<nd);
		};
		/* Set nel values in data[idx_m+i*m] from val[idx_n+i*n] */
		inline void setDataValues(int idx_m, int m, int nel, T *val, int idx_n, int n){	
			for(int i=0;i<nel;i++)	setDataValue(idx_m+i*m,val[idx_n+i*n]);
		};

		/* Accessors */
		inline int getDim(){ return dim; };
		inline int getNumData(){ return nd; };
		inline int getSize(int idx){ 
			if(idx>=dim)	assert(idx<dim);
			return sz[idx];
		}; 
		inline int* getSizes(){ 
			int *sizes = new int[dim];
			for(int i=0;i<dim;i++)
				sizes[i] = sz[i];
			return sizes;
		};
		inline T getDataValue(int idx){	
			if(idx>=nd) 	assert(idx<nd);
			return data[idx];
		};
		/* Get nel values from data[idx_m+i*m] to val[idx_n+i*n] */
		inline void getDataValues(int idx_m, int m, int nel, T *val, int idx_n, int n){	
			for(int i=0;i<nel;i++)	val[idx_n+i*n] = getDataValue(idx_m+i*m); 
		};
		inline T* getData(){
			T *d = new T[nd];
			assert(d);
			for(int i=0;i<nd;i++)
				d[i]=data[i];
			return d;
		};
		inline T* getDataPointer(){	return data; };

		/* I/O */
		/* Read from a binary file */
		void readFromBinary(const char *filename);
		/* Write to a binary file */
		void writeToBinary(const char *filename);

		/* Free memory of sz and data */
		inline void free(){
			delete[] sz;
			delete[] data;
		};
};

/* Escritura a un archivo binario */
template <typename T>
void matrix<T>::writeToBinary(const char *filename){
  FILE *fid;
  fid = fopen(filename,"wb"); 
  assert(fid);
  fwrite(&dim,sizeof(int),1,fid);
  fwrite(sz,sizeof(int),dim,fid);
  fwrite(data,sizeof(T),nd,fid);  
  fclose(fid);
}

/* Lectura desde un archivo binario */
template <typename T>
void matrix<T>::readFromBinary(const char *filename){
  FILE *fid;
  fid = fopen(filename,"rb"); 
  assert(fid);
  fread(&dim,sizeof(int),1,fid);
  sz = new int[dim];
  assert(sz);
  fread(sz,sizeof(int),dim,fid);
  nd = 1;
  for(int i=0;i<dim;i++)
    nd *= sz[i];
  data = new T[nd];
  assert(data);
  fread(data,sizeof(T),nd,fid);  
  fclose(fid);
}
