#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/FSUGarnet
time ../CovEllipse.x << \EOF
 0.15317737, -16.23069416, 0.00048267, 0.02003718, -0.290332742 
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

