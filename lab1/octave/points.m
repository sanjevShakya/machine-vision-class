 P1 = [2;4;2]
 P2 = [6;3;3]
 P3 = [1;2;0.5] 
 P4 = [16;8;4] 

 plot(P1, 'x')
 hold on
 plot(P2, 'x')
 plot(P3, 'x')
 plot(P4, 'x')
 hold off
 l1 = [8,-4,0]
 # plot(l1)
 testP1 = dot(P1, l1)
 testP2 = dot(P2, l1)
 testP3 = dot(P3, l1)
 testP4 = dot(P4, l1)
 
 h_P1 = P1 / P1(3)
 h_P2 = P2 / P2(3)
 h_P3 = P3 / P3(3)
 h_P4 = P4 / P4(3)
 
 x = [0:5]
 y = 2 * x
 plot(x, y, 'r-', h_P1(1), h_P1(2), 'ro', h_P2(1), h_P2(2), 'bx', h_P3(1), h_P3(2), 'ro', h_P4(1), h_P4(2), 'bx')
