#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/RMF032
time ../CovEllipse.x << \EOF
 0.14981891, -16.34635837, 0.00049060, 0.02125960, -0.411680612
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

