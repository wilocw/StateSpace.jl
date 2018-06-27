
syms x y n1 n2 L1 L2 p(x,y,n1,n2)

p(x,y,n1,n2) = sin(pi*n1*(x + L1)/(2*L1))*sin(pi*n2*(y+L2)/(2*L2))/sqrt(L1*L2)

q(x,y,n1,n2) = sin(pi*n1*x/L1)*sin(pi*n2*y/L2)