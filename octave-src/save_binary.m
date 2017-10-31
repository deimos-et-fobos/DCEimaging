function save_binary(x,filename)
	szs=size(x);
	dim=length(szs);
	nx=1;	
	for i=1:dim
		nx = nx*szs(i);
	end	
	fid = fopen(filename,'wb');
	fwrite(fid,dim,'int32');
	fwrite(fid,szs,'int32');
	fwrite(fid,x,'single');
	fclose(fid);
endfunction
