load bio_parameters.mat;
load DCE_MRI.mat;
set(0,'defaultlinelinewidth',3);
sx=size(C_t,1);sy=size(C_t,2);sz=size(C_t,3);st=size(C_t,4);
diff_t=mean(diff(t));
Cp=AIF';
fftCp=fft(Cp,64);
if(size(t,1)<size(t,2))
	t=t';
end

for ii=245:sx
  for jj=300:sy
    for kk=11:11
			Ct=reshape(C_t(ii,jj,kk,:),st,1);
      vp0=vp(ii,jj,kk);ktrans0=ktrans(ii,jj,kk);kep0=kep(ii,jj,kk);
			exponential=exp(-kep0*t);
			ct_convol=real(ifft(fftCp.*fft(exponential,64),64));
			integral=ct_convol(1:30)*diff_t;
			Ct_k=vp0*Cp+ktrans0*integral;
			fprintf('vp=%f\tktrans=%f\tkep=%f\n',vp0,ktrans0,kep0);

      vp0=Vp(ii,jj,kk);ktrans0=Ktrans(ii,jj,kk);kep0=Kep(ii,jj,kk);
      exponential=exp(-kep0*t);
			ct_convol=real(ifft(fftCp.*fft(exponential,64),64));
      integral=ct_convol(1:30)*diff_t;
      Ct_hum=vp0*Cp+ktrans0*integral;
	
			plot(t,Ct,t,Ct_k,t,Ct_hum);legend('Data','Richi','Humberto');axis([0,t(end),-0.1,1]);
%			plot(t,Ct,'x',t,Ct_k);legend('Data','Richi');axis([0,t(end),-0.1,1]);
			title(sprintf('Voxel(%d,%d,%d)',ii,jj,kk));
			pause();
    end
  end
end
