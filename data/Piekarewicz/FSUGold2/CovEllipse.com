#!/bin/csh
cd /Users/jpiekarewicz/SourceCodes/Year2022/Saturation/FSUGold2
time ../CovEllipse.x << \EOF
 0.15060934, -16.28037420, 0.00069031, 0.02070697, -0.286051878 
\EOF
mv fort.8 CovEllipse.out
exit

 FOR RUNNING THE HARTREE CODE:
 -line 1: <A>, <B>, StDevA, StDevA, Corr(A,B)

