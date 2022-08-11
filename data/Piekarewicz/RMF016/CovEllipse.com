#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/RMF016
time ../CovEllipse.x << \EOF
 0.15317940, -16.23045213, 0.00048769, 0.02038334, -0.308589947 
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

