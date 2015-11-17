iterations=5;
uv=zeros(iterations+1,2);
delta_uv=zeros(iterations,2);
Hessian=zeros(2,2);
gradient=zeros(2,1);
for i=2:(iterations+1)
    Hessian(1,1)=(exp(uv(i-1,1))+uv(i-1,2)^2*exp(uv(i-1,1)*uv(i-1,2))+2);
    Hessian(2,1)=exp(uv(i-1,1)*uv(i-1,2))+uv(i-1,1)*uv(i-1,2)*exp(uv(i-1,1)*uv(i-1,2))-2;
    Hessian(1,2)=Hessian(2,1);
    Hessian(2,2)=(4*exp(2*uv(i-1,2))+uv(i-1,1)^2*exp(uv(i-1,1)*uv(i-1,2))+4);
    gradient(1,1)=exp(uv(i-1,1))+uv(i-1,2)*exp(uv(i-1,1)*uv(i-1,2))+2*uv(i-1,1)-2*uv(i-1,2)-3;
    gradient(2,1)=2*exp(2*uv(i-1,2))+uv(i-1,1)*exp(uv(i-1,1)*uv(i-1,2))-2*uv(i-1,1)+4*uv(i-1,2)-2;
    delta_uv(i-1,:)=(-(Hessian)^(-1)*gradient)';
    uv(i,:)=uv(i-1,:)+delta_uv(i-1,:);
end
E=exp(uv(6,1))+exp(2*uv(6,2))+exp(uv(6,1)*uv(6,2))+uv(6,1)^2-2*uv(6,1)*uv(6,2)+2*uv(6,2)^2-3*uv(6,1)-2*uv(6,2)