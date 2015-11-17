step_size=0.01;
iterations=5;
u=zeros(iterations+1,1);
v=zeros(iterations+1,1);
delta_u=zeros(iterations,1);
delta_v=zeros(iterations,1);
for i=2:(iterations+1)
    delta_u(i-1)=exp(u(i-1))+v(i-1)*exp(u(i-1)*v(i-1))+2*u(i-1)-2*v(i-1)-3;
    delta_v(i-1)=2*exp(2*v(i-1))+u(i-1)*exp(u(i-1)*v(i-1))-2*u(i-1)+4*v(i-1)-2;
    u(i)=u(i-1)-step_size*delta_u(i-1);
    v(i)=v(i-1)-step_size*delta_v(i-1);
end
E=exp(u(6))+exp(2*v(6))+exp(u(6)*v(6))+u(6)^2-2*u(6)*v(6)+2*v(6)^2-3*u(6)-2*v(6)