load DCE_MRI.mat;
sx=size(C_t,1);sy=size(C_t,2);sz=size(C_t,3);st=size(C_t,4);
vp=ktrans=kep=converged=zeros(sx,sy,sz);
diff_t=mean(diff(t));
maxit = 1000;
Cp=AIF';
J=zeros(st,3);
integral=dintegral=Ct_k=zeros(st,1);
if(size(t,2)>size(t,1))
	t=t';
end
its=0;
fftCp=fft(Cp,64);

printf('Starting calculation...\n');
fflush(stdout);
tic
for ii=214:sx
	its=0;
	printf('Voxels(%3d,:,:) -> ',ii);
  for jj=270:sy
    for kk=11:11
			% Parámetros y Ct_k iniciales
			printf('Voxels(%d,%d,%d) -> ',ii,jj,kk);
			Ct=reshape(C_t(ii,jj,kk,:),st,1);
      vp0=0.1;ktrans0=0.3/60;kep0=0.3/60;
			exponential=exp(-kep0*t);
			ct_convol=real(ifft(fftCp.*fft(exponential,64),64));
			integral=ct_convol(1:30)*diff_t;
			Ct_k=vp0*Cp+ktrans0*integral;
      dCt=Ct-Ct_k;
			r_old=norm(dCt);

      % Cuadrados mínimos no-lineal
      k=0;dbeta=[1d10,1d10,1d10];dr=r_min=1d10;
      while( k<maxit && (norm(dbeta)>1d-6||dr>1d-3) && kep0>1d-9)
				k++;
        dct_convol=real(ifft(fftCp.*fft(exponential.*t,64),64));
        dintegral=dct_convol(1:30)*diff_t;
        J(:,1)=Cp;J(:,2)=integral;J(:,3)=-ktrans0*dintegral;
        dbeta=(J'*J)\(J'*dCt);
				if(vp0+dbeta(1)<0) dbeta(1)=-vp0*3/4; endif
				if(ktrans0+dbeta(2)<0) dbeta(2)=-ktrans0*3/4; endif
				if(kep0+dbeta(3)<0) dbeta(3)=-kep0*3/4; endif
        vp0=vp0+dbeta(1);ktrans0=ktrans0+dbeta(2);kep0=kep0+dbeta(3);
				beta=[vp0,ktrans0,kep0];
        exponential=exp(-kep0*t);
        ct_convol=real(ifft(fftCp.*fft(exponential,64),64));
        integral=ct_convol(1:30)*diff_t;
        Ct_k=vp0*Cp+ktrans0*integral;
        dCt=Ct-Ct_k;
				r_new=norm(dCt);
				if(r_new<r_min)	r_min=r_new; beta_min=beta; end
				dr=abs(r_old-r_new)/r_new;
%				[dr r_new norm(dbeta) beta]
%				[dr r_old r_new norm(dbeta) beta (k<maxit && (norm(dbeta)>1d-6||dr>1d-6) && kep0>-0.1)]
%				plot(t,Ct,t,Ct_k,t,Ct_hum);legend('Data','Richi','Humberto');axis([0,t(end),-0.1,1]);
%				pause();
				r_old=r_new;
      end
			if(k==maxit)
%				printf('Voxel(%d,%d,%d) no convergido.\n',ii,jj,kk);
%				fflush(stdout);
			else
				converged(ii,jj,kk)=1;
			end
			vp(ii,jj,kk)=beta_min(1);
			ktrans(ii,jj,kk)=beta_min(2);
			kep(ii,jj,kk)=beta_min(3);
			if(k>its && k<maxit)
				its = k;
			end
    end
  end
%	save '-mat7-binary' bio_params.mat vp ktrans kep converged
%	if(mod(ii,25)==0)
%		save '-mat7-binary' bio_params.mat vp ktrans kep converged
%	end
	printf('max.its = %4d -> %f seconds\n', its,toc);
	fflush(stdout);
end
printf('Finished in %f seconds...\n',toc);
printf('Saving...\n');
fflush(stdout);
%save '-mat7-binary' bio_params.mat vp ktrans kep converged
