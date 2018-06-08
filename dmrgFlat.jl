function dosvdtrunc(AA,m)		# AA a matrix;  keep at most m states
  (u,d,v) = svd(AA)
  prob = dot(d,d)		# total probability
  mm = min(m,length(d))	# number of states to keep
  d = d[1:mm]			# middle matrix in vector form
  trunc = prob - dot(d,d)
  U = u[:,1:mm]
  V = v[:,1:mm]'
  (U,d,V,trunc)		# AA == U * diagm(d) * V	with error trunc
end

function dosvdleftright(AA,m,toright)
  (U,d,V,trunc) = dosvdtrunc(AA,m)
  if toright
    V = diagm(d) * V
  else
    U = U * diagm(d)
  end
  (U,V,trunc)
end

function dosvd6(AA,m,toright)	# AA is ia * 2 * 2 * ib;  svd down the middle;  return two parts
  ia = size(AA,1)
  ib = size(AA,6)
  AA = reshape(AA,ia*4,4*ib)
  (U,V,trunc) = dosvdleftright(AA,m,toright)
  mm = size(U,2)
  U = reshape(U,ia,2,2,mm)
  V = reshape(V,mm,2,2,ib)
  (U,V,trunc)
end

using TensorOperations

function JK(a,b)	# Julia kron,  ordered for julia arrays; returns matrix
  (a1,a2) = size(a)
  (b1,b2) = size(b)
  reshape(Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2],a1*b1,a2*b2)
end

function JK4(a,b)	# Julia kron,  ordered for julia arrays, return expanded into 4 indices
  (a1,a2) = size(a)
  (b1,b2) = size(b)
  Float64[a[i,ip] * b[j,jp] for i=1:a1, j=1:b1, ip=1:a2, jp=1:b2]
end

sz = Float64[0.5 0; 0 -0.5]
sp = Float64[0 1; 0 0]
sm = sp'
Htwosite = reshape(JK(sz,sz) + 0.5 * JK(sp,sm) + 0.5 * JK(sm,sp),2,2,2,2)
# order for Htwosite is s1, s2, s1p, s2p

n = 14		# exact n=28 energy is -12.2254405486
#  Make initial product state in up down up down up down pattern (Neel state)
# Make first tensor a 1 x 2 x m tensor; and last is m x 2 x 1  (rather than vectors)
A = [zeros(1,2,2,1) for i=1:n]
for i=1:n
  A[i][1,iseven(i) ? 2 : 1,iseven(i) ? 1 : 2,1] = 1.0
end

HLR = [zeros(1,1) for i=1:n]	# Initialize to avoid errors on firs sweep
ms = 3
for swp = 0:1
  ms = round(Int64,1.3*ms)
  m = ms*ms
  for ii=-n+1:n-1		# if negative, going right to left
    ii == 0 && continue
    i = abs(ii)
    toright = ii > 0

    println("\n sweep, i, dir, m = $swp, $i, ",toright ? "to right" : "to left"," $m")

    dleft = size(A[i],1)
    alpha = dleft * 4
    dright = size(A[i+1],4)
    beta = 4 * dright

    HL = zeros(dleft,2,2,dleft,2,2)
    HR = zeros(2,2,dright,2,2,dright)
    if i > 1
      Aim1 = A[i-1]
      @tensor begin
        HL[a,sit,sib,ap,sitp,sibp] := Htwosite[sim1t,sit,sim1tp,sitp] * Htwosite[sim1b,sib,sim1bp,sibp] * Aim1[b,sim1t,sim1b,a] * Aim1[b,sim1tp,sim1bp,ap]
      end
    end
    HL = reshape(HL,alpha,alpha)
    i > 1 && (HL += JK(HLR[i-1],eye(4)))
    if i < n-1
      Ai2 = A[i+2]
      @tensor begin
        HR[si1t,si1b,b,si1tp,si1bp,bp] := Htwosite[si1t,si2t,si1tp,si2tp] * Htwosite[si1b,si2b,si1bp,si2bp] * Ai2[b,si2t,si2b,a] * Ai2[bp,si2tp,si2bp,a]
      end
    end
    HR = reshape(HR,beta,beta)
    i < n-1 && (HR += JK(eye(4),HLR[i+2]))

    OleftT =  Any[JK(JK(eye(dleft),sz),eye(2)), 0.5*JK(JK(eye(dleft),sp),eye(2)), 0.5*JK(JK(eye(dleft),sm),eye(2))]
    OrightT = Any[JK(sz,eye(2*dright)),JK(sm,eye(2*dright)),JK(sp,eye(2*dright))]

    OleftB =  Any[JK(eye(dleft*2),sz), 0.5*JK(eye(dleft*2),sp), 0.5*JK(eye(dleft*2),sm)]
    OrightB = Any[JK(eye(2),JK(sz,eye(dright))),JK(eye(2),JK(sm,eye(dright))),JK(eye(2),JK(sp,eye(dright)))]

    Ai = A[i]
    Ai1 = A[i+1]
    @tensor begin
      AA[a,bt,bb,dt,db,e] := Ai[a,bt,bb,c] * Ai1[c,dt,db,e]
    end

    #  Inefficient implementation:  m^4   Ham construction
    Ham = zeros(alpha*beta,alpha*beta)
    for j=1:length(OleftT)
      Ham += JK(reshape(OleftT[j],alpha,alpha),reshape(OrightT[j],beta,beta))
      Ham += JK(reshape(OleftB[j],alpha,alpha),reshape(OrightB[j],beta,beta))
    end


    if i > 1
      Ham += JK(HL,eye(beta))
    end
    if i < n-1
      Ham += JK(eye(alpha),HR)
    end
    if i == 1
      Ham += JK(JK(eye(dleft),reshape(Htwosite,4,4)),eye(beta))
    end
    if i == n-1
      Ham += JK(eye(alpha),JK(reshape(Htwosite,4,4),eye(dright)))
    end



    bigH = reshape(Ham,alpha*beta,alpha*beta)
    bigH = 0.5 * (bigH + bigH')
    evn = eigs(bigH;nev=1, which=:SR,ritzvec=true,v0=reshape(AA,alpha*beta))
    @show evn[1]
    @show size(evn[2])
    gr = evn[2][:,1]

    #test
    Ham = zeros(alpha*beta,alpha*beta)
      if i < n-1
        Ham += JK(eye(alpha),HR)
      end
      if i == n-1
        Ham += JK(eye(alpha),JK(reshape(Htwosite,4,4),eye(dright)))
      end
    rightE = gr'*Ham*gr
    println("Right Energy = $rightE")

    #test
    Ham = zeros(alpha*beta,alpha*beta)
      if i > 1
        Ham += JK(HL,eye(beta))
      end
      if i == 1
        Ham += JK(JK(eye(dleft),reshape(Htwosite,4,4)),eye(beta))
      end
    leftE = gr'*Ham*gr
    println("Left Energy = $leftE")

    #test
    Ham = zeros(alpha*beta,alpha*beta)
    for j=1:length(OleftT)
      Ham += JK(reshape(OleftT[j],alpha,alpha),reshape(OrightT[j],beta,beta))
      Ham += JK(reshape(OleftB[j],alpha,alpha),reshape(OrightB[j],beta,beta))
    end
    centerE = gr'*Ham*gr
    println("Center Energy = $centerE")

    AA = reshape(gr,dleft,2,2,2,2,dright)
    (A[i],A[i+1],trunc) = dosvd6(AA,m,toright)
    @show trunc
    if toright && i < n-1
      if 1 < i
        (i1,i2,i3,i4) = size(A[i])
        Ai2 = reshape(A[i],i1*i2*i3,i4)
        @tensor begin
          hlri[b,bp] := HL[a,ap] * Ai2[a,b] * Ai2[ap,bp]
        end
      elseif i == 1
        Ai = A[i]
        @tensor begin
          hlri[c,cp] := Htwosite[a,b,ap,bp] * Ai[e,a,b,c] * Ai[e,ap,bp,cp]
        end
      end
      HLR[i] = hlri
    elseif !toright && 1 < i
      if i < n-1
        (i1,i2,i3,i4) = size(A[i+1])
        Ai12 = reshape(A[i+1],i1,i2*i3*i4)
        @tensor begin
          hlri1[a,ap] := HR[b,bp] * Ai12[a,b] * Ai12[ap,bp]
        end
      elseif i == n-1
        Aip1 = A[i+1]
        @tensor begin
          hlri1[e,ep] := Htwosite[a,b,ap,bp] * Aip1[e,a,b,c] * Aip1[ep,ap,bp,c]
        end
      end
      HLR[i+1] = hlri1
    end
  end
end
