load signals.dat;
kep=2/5;
ktrans=0.6/5;
ct_trapz1=zeros(size(Cp));
ct_trapz2=zeros(size(Cp));
t=[0:length(t)-1]';
for i=2:length(t)
	tf=t(i);
	tau=t(1:i);
	exponential=exp(-kep*(tf-tau));
  ct_trapz1(i)=trapz(tau,Cp(1:i).*exponential);
	exponential=exp(-kep*tau);
  ct_trapz2(i)=trapz(tau,Cp(i:-1:1).*exponential);
end
exponential=exp(-kep*t);
ct_convol1=real(ifft(fft(Cp,64).*fft(exponential,64),64));
ct_convol2=fftconv(Cp,exponential);
plot(1:length(ct_convol1),ct_convol1,1:length(ct_convol2),ct_convol2);
ct_convol1=ct_convol1(1:length(t));
ct_convol2=ct_convol2(1:length(t));
plot(t,ct_trapz1,t,ct_trapz2,t,ct_convol1,t,ct_convol2);
legend('Ct_trapz1','Ct_trapz2','Ct_convol1','Ct_convol2');

dct_trapz1=zeros(size(Cp));
dct_trapz2=zeros(size(Cp));
t=[0:length(t)-1]';
for i=2:length(t)
	tf=t(i);
	tau=t(1:i);
	exponential=exp(-kep*(tf-tau));
  dct_trapz1(i)=trapz(tau,Cp(1:i).*(tf-tau).*exponential);
	exponential=exp(-kep*tau);
  dct_trapz2(i)=trapz(tau,Cp(i:-1:1).*exponential.*tau);
end
exponential=exp(-kep*t);
dct_convol1=real(ifft(fft(Cp,64).*(fft(exponential.*t,64)),64)); %bien
dct_convol2=fftconv(Cp,exponential.*t);	%bien
plot(1:length(dct_convol1),dct_convol1,1:length(dct_convol2),dct_convol2);
dct_convol1=dct_convol1(1:length(t));
dct_convol2=dct_convol2(1:length(t));
plot(t,dct_trapz1,t,dct_trapz2,t,dct_convol1,t,dct_convol2);
legend('dCt_trapz1','dCt_trapz2','dCt_convol1','dCt_convol2');

plot(t,ct_trapz1,t,ct_trapz2,t,dct_trapz1,t,dct_trapz2);
legend('Ct_trapz1','Ct_trapz2','dCt_trapz1','dCt_trapz2');
