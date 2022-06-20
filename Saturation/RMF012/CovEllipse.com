#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/RMF012
time ../CovEllipse.x << \EOF
 0.15425317, -16.22350914, 0.00050130, 0.01999912, -0.29158285 
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

