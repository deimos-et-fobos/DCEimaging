function x=read_binary(filename)
	fid = fopen(filename,'rb');
	dim = fread(fid,1,'int32');
	szs = fread(fid,dim,'int32');
	ndata = prod(szs);
	x = fread(fid,ndata,'single');
	x = reshape(x,szs');
	fclose(fid);
end
