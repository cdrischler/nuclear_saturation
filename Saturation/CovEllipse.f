c-----------------------------------------------------------------------
c     In this program we compute the covariant ellipse between two 
c     observables "A" and "B". The parameters of the ellipse (a,b,
c     and theta) are computed in the subroutine ellipse from the
c     variance in A and B, and its correlation coefficient.
c                                            June 13, 2014.
c                        Slightly Modified on: May 8, 2022.
c-----------------------------------------------------------------------
      program CovEllipse
c
      implicit real*8 (a-h,j-z)
c
c     Enter averages, variances and correlation coefficient for
c     the two observables.
c     NOTE: multiplying by 1 sA and sB gives the 39% confidence
c     ellipse, whereas multiplying by 2.4475 gives the 95% one.
c
      read(*,*) A0,B0,sA,sB,rAB
      call Ellipse(1.0000*sA,1.0000*sB,rAB,a1,b1,tht1) !parameters of 39% ellipse      
      call Ellipse(2.4475*sA,2.4475*sB,rAB,a2,b2,tht2) !parameters of 39% ellipse 
c     
c     Define the phi-angle grid:
      pi=4.0d0*atan(1.0d0)            !3.1415926...
      iPmax=500                       !number of points
      pmin=0.00000d0                  !minimum phi 
      pmax=2.00000d0*pi               !maximum phi (2*pi)
      dp=(pmax-pmin)/iPmax            !step size in phi
c
c     Write parameters of the ellipse:
      theta1=tht1*180.0d0/pi          !orientation angle (degrees)
      theta2=tht2*180.0d0/pi          !orientation angle (degrees)
      write(8,100) A0,sA,B0,sB,rAB,   !write to file
     .   a1,b1,a2,b2,theta1,theta2    !write to file          
c
c     Generate the 39% and 95% tilted ellipses:
      write(8,200)                    !write to file
      do ip=0,iPmax                   !loop over phi
       phi=pmin+ip*dp                 !select phi   
       x1=a1*cos(phi)*cos(tht1)-b1*sin(phi)*sin(tht1) !value of x1 
       y1=a1*cos(phi)*sin(tht1)+b1*sin(phi)*cos(tht1) !value of y1
       x2=a2*cos(phi)*cos(tht2)-b2*sin(phi)*sin(tht2) !value of x2 
       y2=a2*cos(phi)*sin(tht2)+b2*sin(phi)*cos(tht2) !value of y2
       As1=A0+x1                        !properly centered
       Bs1=B0+y1                        !properly centered
       As2=A0+x2                        !properly centered
       Bs2=B0+y2                        !properly centered
       write(8,300) As1,Bs1,As2,Bs2     !write to file
      end do
c
 100  format('#',2x,'  <A>=',f10.6,2x,'sigmaA=',f9.6,/,
     .       '#',2x,'  <B>=',f10.6,2x,'sigmaB=',f9.6,/,
     .       '#',2x,'rhoAB=',f10.6,/,
     .       '#',2x,'  a39=',f10.6,2x,'   b39=',f10.6,/,
     .       '#',2x,'  a95=',f10.6,2x,'   b95=',f10.6,/,
     .       '#',2x,'tht39=',f10.6,2x,' tht95=',f10.6)
 200  format('#',/,'#',7x,'A39',7x,'B39',9x,'A95',7x,'B95')
 300  format(2x,4(' ',f10.6))
c
      stop
      end
c-----------------------------------------------------------------------
c     Given the St.dev. "sA" and "sB" in two observables A and B
c     and its correlation coefficient, compute the parameters of 
c     their covariance ellipse: "a" semi-major axis, "b" semi-minor 
c     axis, and "theta" the orientation of the ellipse (see notes).
c 
      subroutine Ellipse(sA,sB,rAB,a,b,theta) 
c
      implicit real*8 (a-h,j-z)
c
c     Compute parameters of the ellipse:
      xi=(2*rAB*sA*sB)/(sA**2-sB**2)     !xi-parameter
      eta=sqrt(1.0d0+xi**2)              !eta-parameter
      tantheta=xi/(1+eta)                !tan(theta)
      sumAB=0.50d0*(sA**2+sB**2)         !useful quantity
      difAB=0.50d0*(sA**2-sB**2)         !useful quantity
      a=sqrt(sumAB+difAB*eta)            !semi-major axis
      b=sqrt(sumAB-difAB*eta)            !semi-minor axis
      theta=atan(tantheta)               !theta (in radians)
c
      return
      end
c-----------------------------------------------------------------------
