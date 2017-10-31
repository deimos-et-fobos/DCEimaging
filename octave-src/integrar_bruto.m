%load signals.dat;
kep=2/60;
ktrans=0.6/60;
vp=2/60;
ct_trapz1=zeros(size(Cp));
ct_trapz2=zeros(size(Cp));
for i=2:30
	tf=t(i);
	tau=t(1:i);
	exponential=exp(-kep*(tf-tau));
  ct_trapz1(i)=trapz(tau,Cp(1:i).*exponential);
	exponential=exp(-kep*tau);
  ct_trapz2(i)=trapz(tau,Cp(i:-1:1).*exponential);
end
exponential=exp(-kep*t);
ct_convol=real(ifft(fft(Cp).*fft(exponential),64));
%ct_convol=fftconv(Cp,exponential);
ct_convol=ct_convol(1:30)*mean(diff(t));
plot(t,vp*Cp+ktrans*ct_trapz1,t,vp*Cp+ktrans*ct_trapz2,t,vp*Cp+ktrans*ct_convol,t,Ct);
legend('Ct_trapz1','Ct_trapz2','Ct_convol','Ct_real');
