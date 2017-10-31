load signals.dat;
set(0,'defaultlinelinewidth',3);
kep1=kep2=0.035;
ktrans1=ktrans2=0.011;
vp1=vp2=-0.1;
J1=J2=zeros(size(Cp,1),3);
integral1=integral2=zeros(size(Cp));
dintegral1=dintegral2=zeros(size(Cp));
Ct_k1=Ct_k2=zeros(size(Ct));

% Cuadrados mínimos no-lineal
k=0;dbeta1=dbeta2=1;
while( k<1000 && (norm(dbeta1)>1d-6 || norm(dbeta2)>1d-6) )
	printf('\n\nk=%f\t abs(dbeta1)=%f\t abs(dbeta2)=%f\n',++k,norm(dbeta1),norm(dbeta2));
	for i=2:30
		tf=t(i);
		tau=t(1:i);
		exponential=exp(-kep1*(tf-tau));
  	integral1(i)=trapz(tau,Cp(1:i).*exponential);
		exponential=exp(-kep1*tau);
	  dintegral1(i)=trapz(tau,Cp(i:-1:1).*exponential.*tau);
	end
	exponential=exp(-kep2*t);
	ct_convol=fftconv(Cp,exponential);
	integral2=ct_convol(1:30)*mean(diff(t));
	dct_convol=fftconv(Cp,exponential.*t);
	dintegral2=dct_convol(1:30)*mean(diff(t));
	J1(:,1)=Cp;J1(:,2)=integral1;J1(:,3)=-ktrans1*dintegral1;
	J2(:,1)=Cp;J2(:,2)=integral2;J2(:,3)=-ktrans2*dintegral2;
	Ct_k1=vp1*Cp+ktrans1*integral1;
  dCt=Ct-Ct_k1;	
	dbeta1=(J1'*J1)\(J1'*dCt);
	vp1=vp1+dbeta1(1);ktrans1=ktrans1+dbeta1(2);kep1=kep1+dbeta1(3);
	Ct_k2=vp2*Cp+ktrans2*integral2;
  dCt=Ct-Ct_k2;	
	dbeta2=(J2'*J2)\(J2'*dCt);
	vp2=vp2+dbeta2(1);ktrans2=ktrans2+dbeta2(2);kep2=kep2+dbeta2(3);
	printf('dbeta1: %f %f %f.\t vp1=%f\t ktrans1=%f\t kep1=%f\n',dbeta1,vp1,ktrans1,kep1);
	printf('dbeta2: %f %f %f.\t vp2=%f\t ktrans2=%f\t kep2=%f\n',dbeta2,vp2,ktrans2,kep2);
	beta1=[vp1,ktrans1,kep1];
	beta2=[vp2,ktrans2,kep2];
end
plot(t,Ct,t,Ct_k1,t,Ct_k2);
legend('Ct_{real}','Ct_{integ}','Ct_{convol}');

% Variación de parámetros
vp=-0.084:0.0001:-0.082;
ktrans=0.0094:0.00001:0.0096;
kep=0.0258:0.00001:0.0260;
r_min=1d10;
diff_t=mean(diff(t));
for i=1:length(vp)
	for j=1:length(ktrans)
		for k=1:length(kep)
			exponential=exp(-kep(k)*t);
  		ct_convol=fftconv(Cp,exponential);
  		integral=ct_convol(1:30)*diff_t;
			Ct_n = vp(i)*Cp+ktrans(j)*integral;
			r = norm(Ct-Ct_n);
			if(r<r_min)
				r_min=r;
				beta=[vp(i),ktrans(j),kep(k)];
				Ct_min = Ct_n;
			end
		end
	end
end

plot(t,Ct,t,Ct_k1,t,Ct_k2,t,Ct_min);
legend('Ct_{real}','Ct_{integ}','Ct_{convol}','Ct_{param}');
