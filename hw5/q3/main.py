#python 2
from svmutil import *
y, x=[-1,-1,-1,1,1,1,1],[[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]
prob=svm_problem(y,x)
param=svm_parameter('-t 1 -g 1 -d 2 -r 1')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
print 'support_vector_coefficients='
print support_vector_coefficients
print 'note that these coefficients are aplha_n*y_n'
support_vector=m.get_SV()
print 'support_vector='
print support_vector
support_vector_index=m.get_sv_indices()
print 'support_vector_indices='
print support_vector_index