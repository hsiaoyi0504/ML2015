%%for N=5
xMin=2.5;
xMax=7.5;
yMin=0;
yMax=20;
%%for N=10000
% xMin=9997.5;
% xMax=10002.5;
% yMin=0;
% yMax=1;

h1=ezplot('sqrt( 8 / x * log( 4 * ( ( 2 * x ) ^ 50 ) / 0.05 ) )',[xMin xMax yMin yMax]);
hold on;
h2=ezplot('sqrt( 16 / x * log( 2 * ( ( x ) ^ 50 ) / sqrt(0.05) ) )',[xMin xMax yMin yMax]);
h3=ezplot('sqrt( 2 * log( 2 * x * ( ( x ) ^ 50 ) ) / x ) + sqrt( 2 / x * log( 1 / 0.05 ) ) + 1 / x ',[xMin xMax yMin yMax]);
h4=ezplot('y - sqrt( 1 / x * ( 2 * y + log( 6 * ( (2 * x ) ^ 50 ) / 0.05 ) ) )',[xMin xMax yMin yMax]);
%h5=ezplot('y - sqrt( ( 1 / ( 2 * x ) ) * ( 4 * y * ( 1 + y ) + log( 4 * ( (x ^2 ) ^ 50 ) / 0.05  ) ) )',[xMin xMax yMin yMax]);
h5=ezplot('y - sqrt( ( 1 / ( 2 * x ) ) * ( 4 * y * ( 1 + y ) + log( 4 ) + 100 * log( x ) - log( 0.05  ) ) )',[xMin xMax yMin yMax]);
hold off;
xlabel('N');
ylabel('Epsilon');
title('Comparison of bounds');
legend('Original VC bound','Variant VC bound','Rademacher Penalty Bound','Parrondo and Van den Broek','Devroye');
set(h1,'color','r','linestyle','-.')
set(h2,'color','g','linestyle','-.')
set(h3,'color','b','linestyle','-.')
set(h4,'color','c','linestyle','--')
set(h5,'color','m','linestyle','--')