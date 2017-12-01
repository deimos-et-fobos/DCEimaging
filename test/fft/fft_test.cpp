#include <stdio.h>
#include <math.h>
#include <vector>
#include "../../src/FFT_CODE/fft.h"
#include "../../src/FFT_CODE/complex.h"

#define NX 64
#define ndata 30

int main(void){
	int dummy;
	FILE* fAIF;
	complex *Cp,*fftCp,*ifftCp;
	double *AIF,*fftAIF_re,*fftAIF_im,*ifftAIF; 
	AIF = new double[NX];
	fftAIF_re = new double[NX];
	fftAIF_im = new double[NX];
	ifftAIF = new double[NX];
	Cp = new complex[NX];
	fftCp = new complex[NX];
	ifftCp = new complex[NX];

	/* Leo los datos y los guardo en Cp */
	fAIF = fopen("AIF.dat","r"); 
	for(int i=0;i<NX;i++)
		fscanf(fAIF,"%lf%lf%lf%lf",&AIF[i],&fftAIF_re[i],&fftAIF_im[i],&ifftAIF[i]);
	fclose(fAIF);

	/* Copio Cp a un arreglo de complex */
	for(int i=0;i<NX;i++)
		Cp[i] = AIF[i];

	/* Realizo la fft y la ifft */
	CFFT::Forward(Cp,fftCp,NX);
	CFFT::Inverse(fftCp,ifftCp,NX);

	/* Imprimo */
	printf("Cp\n");
	for(int i=0;i<NX;i++)
		printf("%f -> %f\n",AIF[i],Cp[i]);	
	printf("fft(Cp)\n");
	for(int i=0;i<NX;i++)
		printf("(%f,%f) -> (%f,%f)\n",fftAIF_re[i],fftAIF_im[i],fftCp[i].re(),fftCp[i].im());	
	printf("ifft(Cp)\n");
	for(int i=0;i<NX;i++)
		printf("%f -> %f\n",ifftAIF[i],ifftCp[i].re());	
	
	/* Realizo la fft y la ifft in-place*/
	CFFT::Forward(Cp,NX);
	printf("fft(Cp)\n");
	for(int i=0;i<NX;i++)
		printf("(%f,%f) -> (%f,%f)\n",fftAIF_re[i],fftAIF_im[i],fftCp[i].re(),fftCp[i].im());	
	CFFT::Inverse(Cp,NX);
	printf("ifft(Cp)\n");
	for(int i=0;i<NX;i++)
		printf("%f -> %f\n",AIF[i],Cp[i]);	
	
	delete[] AIF;
	delete[] fftAIF_re;
	delete[] fftAIF_im;
	delete[] ifftAIF;
	delete[] Cp;
	delete[] fftCp;
	delete[] ifftCp;

	return 0;
}
