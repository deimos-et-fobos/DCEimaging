/**********************************************/
/* TEST: fft and ifft of f(t) = A*exp(-a*t)*t */
/**********************************************/

#include <stdio.h>
#include <math.h>
#include <vector>
#include "../../src/FFT_CODE/fft.h"
#include "../../src/FFT_CODE/complex.h"

#define NX 4096
#define dt 0.001
#define tau 5.0
#define ndata 30

int main(void){
	int i;
	complex data_complex[NX],fft_data[NX],ifft_data[NX];
	double data[NX]; 

	for(i=0;i<NX;i++){
		data[i]=2.0*exp(-tau*i*dt)*i*dt;
		printf("%f\n",data[i]);
	}
	
	/* Copio la señal a un arreglo de complex */
	for(int i=0;i<NX;i++){
		data_complex[i] = data[i];
		printf("%f\t%f\n",data_complex[i].re(),data_complex[i].im());
	}

	/* Realizo la fft y la ifft */
	CFFT::Forward(data_complex,fft_data,NX);
	CFFT::Inverse(fft_data,ifft_data,NX);

	/* Imprimo */
	printf("Data\tfft\t\tifft\n");
	for(int i=0;i<NX;i++)
		printf("%f\t%f\t%f\t%f\t%f\n",data[i],fft_data[i].re(),fft_data[i].im(),ifft_data[i].re(),ifft_data[i].im());
	
	return 0;
}
