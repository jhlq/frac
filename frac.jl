useGPU=false
if useGPU==true
	import OpenCL
	const cl = OpenCL
end

const maxiter=255
const dim=1000
const maplim=1.8126796e6
#map=readdlm("map.csv",',')
records=readdlm("records.csv",',')
#record=records[end,4]
#toplocs=Array(Array{Float64,1},0)#[-1,-1,1]
#push!(toplocs,[-1.0,-1.0,1,1.7378675e6]) #x,y,zoom,inequality

function mandel(z)
	c = z #complex64(-0.5, 0.75)
	for n = 1:maxiter
		if abs2(z) > 4.0
			return n-1
		end
		z = z*z + c
	end
	return maxiter
end
#w=1
#h=1
#z=1
#x=-1
#y=-1
#q = [complex64(r,i) for i=linspace(x,x+1/z,w), r=linspace(y,y+1/z,h)];
#y:-(2.0/z/w):-y
#-1.5:(3.0/h):1.5
function ordinary_mandel(q)
	(h, w) = size(q)
	m  = Array(Uint8, (h, w));
	for i in 1:w
		for j in 1:h
			@inbounds v = q[j, i]
			@inbounds m[j, i] = mandel(v)
		end
	end
	return m
end
function mandel_cpu(x::Float64,y::Float64,z::Float64,w::Int64=dim,h::Int64=dim)
	q = [complex64(r,i) for i=linspace(y,y+1/z,h), r=linspace(x,x+1/z,w)];
	m  = Array(Uint8, (h, w));
	for i in 1:w
		for j in 1:h
			@inbounds v = q[j, i]
			@inbounds m[j, i] = mandel(v)
		end
	end
	return m
end
mandel_cpu(x::Real,y::Real,z::Real)=mandel_cpu(float(x),float(y),float(z))
mandel_cpu(a::Array)=mandel_cpu(a[1],a[2],a[3])
function inequality(m::Array{Uint8,2})
	itspace=zeros(Int64,maxiter+1)
	for pix in m
		itspace[int(pix)+1]+=1
	end
	ism=mean(itspace)
	sco=0
	for it in itspace
		sco+=abs(it-ism)
	end
	return sco
end
if useGPU==true
	mandel_source = "
	#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
	__kernel void mandelbrot(__global float2 *q,
						 __global ushort *output, 
						 ushort const maxiter)
	{
	 int gid = get_global_id(0);
	 float nreal, real = 0;
	 float imag = 0;
	 output[gid] = 0;
	 for(int curiter = 0; curiter < maxiter; curiter++) {
		 nreal = real*real - imag*imag + q[gid].x;
		 imag = 2* real*imag + q[gid].y;
		 real = nreal;

		 if (real*real + imag*imag > 4.0f)
		 output[gid] = curiter;
	  }
	}";

	function mandel_opencl(q::Array{Complex64}, maxiter::Int64)
		ctx   = cl.Context(cl.devices()[4])
		queue = cl.CmdQueue(ctx)

		out = Array(Uint16, size(q))

		q_buff = cl.Buffer(Complex64, ctx, (:r, :copy), hostbuf=q)
		o_buff = cl.Buffer(Uint16, ctx, :w, length(out))

		prg = cl.Program(ctx, source=mandel_source) |> cl.build!
		
		k = cl.Kernel(prg, "mandelbrot")
		cl.call(queue, k, length(out), nothing, q_buff, o_buff, uint16(maxiter))
		cl.copy!(queue, out, o_buff)
		
		return out
	end
	function mandel_gpu(x::Float64,y::Float64,z::Float64,w::Int64=dim,h::Int64=dim)
		q = [complex64(r,i) for i=linspace(y,y+1/z,h), r=linspace(x,x+1/z,w)];
		m = mandel_opencl(q,maxiter)		
		return m
	end
end
function dig(n::Int64)
	map=readdlm("map.csv",',')
	records=readdlm("records.csv",',')
	record=records[end,4]
	m=0
	for it in 1:n
		loci=rand(1:size(map,1))
		(x,y,z,s)=(map[loci,1],map[loci,2],map[loci,3],map[loci,4])
		nx=x+rand()/z
		ny=y+rand()/z
		nz=10z
		if useGPU==true
			m=mandel_gpu(nx,ny,nz)
		else
			m=mandel_cpu(nx,ny,nz)
		end
		ns=inequality(m)
		if ns<maplim
			addtomap(nx,ny,nz,ns)
			map=cat(1,map,[nx ny nz ns])
		end
		if ns<record
			addrecord(nx,ny,nz,ns)
#			push!(toplocs,[nx,ny,nz,ns])
			println("New record! $ns at: $nx $ny $nz")
			records=cat(1,records,[nx ny nz ns])
			#return m
			record=ns
		end
		print("$ns, ")
	end
	#return m
end
function dig(a::Array{Bool,1}=[true],batchsize::Int64=1000)
	while a[1]==true
		dig(batchsize)
	end
end
function addtomap(x,y,z,inq)
	fh=open("map.csv","a")
	write(fh,"$x,$y,$z,$inq\n")
	close(fh)
end
function addrecord(x,y,z,inq)
	fh=open("records.csv","a")
	write(fh,"$x,$y,$z,$inq\n")
	close(fh)
end
