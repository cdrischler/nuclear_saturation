#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/RMF022
time ../CovEllipse.x << \EOF
 0.15178061, -16.24908255, 0.00046273, 0.02118417, -0.375325077
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

