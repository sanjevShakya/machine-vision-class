A = [1 1 1; 1 1 1 ; 1 1 1]
rank(A)
[u, d, v] = svd (A)
u_t = u'
d0 = [1/d(1,1) 0 0; 0 0 0; 0 0 0]
B = v * d0 * u_t