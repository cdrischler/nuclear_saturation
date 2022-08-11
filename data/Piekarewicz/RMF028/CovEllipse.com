#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/RMF028
time ../CovEllipse.x << \EOF
 0.15128289, -16.29113016, 0.00049995, 0.02041218, -0.441671888
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

