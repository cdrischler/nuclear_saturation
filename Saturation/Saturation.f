c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     This program assumes that a file "Sampling.out" containing
c     10,000 models distributed according to a given covariance
c     matrix obtained from an accurately calibration procedure
c     already exists. All one does here is essentially trade the
c     Fermi momentum for the density and recompute all relevant
c     statistical quantities. Although more general, the aim here
c     is to compute and correlate the saturation density and the
c     binding energy per nucleon at saturatio; hence the name.
c                                                 May 7, 2022.      
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
      Program Saturation
c
      call Reader
      call Covariances
      call Writer
c     
      stop
      end
c-----------------------------------------------------------------------
      subroutine Reader
c
      implicit real*8 (a-h,j-z) 
      common/Configs/X0(8,10000)
      character*50 header 
c
c     Open unit containing the distribution of models:
      open(unit=3,file="Sampling.out",status="old")
c
c     Read the header in the Sampling.out file:
      do ih=1,24                      !loop over 24 lines
       read(3,'(a)') header           !read the header
      end do                          !close ih-loop
c
c     Define the distribution of models to be used: 
      iCmin=00001                     !first configuration      
      iCmax=10000                     !last  configuration  
      pi=4*atan(1.0d0)                !3.141592653589793238
      do ic=iCmin,iCmax               !loop over configurations  
       read(3,*) ic0,X0(:,ic)         !read bulk properties
       kF= X0(2,ic)                   !kFermi        
       X0(2,ic)=(2*kF**3)/(3*pi**2)   !kFermi to Density
      end do                          !close ic-loop
c
      return
      end
c-----------------------------------------------------------------------
c     Compute Covariances and Correlations
c      
      subroutine Covariances
c
      implicit real*8 (a-h,j-z) 
      parameter (iNmax=10000)
      dimension A(iNmax),B(iNmax)
      common/Configs/X0(8,iNmax)
      common/X0Covariances/Covs(8,8),Corrs(8,8)
c
c     Compute Covariances:
      do ib=1,8                    !loop over ib
       B(:)=X0(ib,:)               !extract B  
       do ia=1,8                   !loop over ia
        A(:)=X0(ia,:)              !extract A 
        call Covar(A,B,Cab,iNmax)  !compute covariance
        Covs(ia,ib)=Cab            !store Cov(A,B)
       end do                      !close ia-loop
      end do                       !close ib-loop
c
c     Compute Correlations:
      do ib=1,8                    !loop over ib
       dB=Covs(ib,ib)              !extract dB  
       do ia=1,8                   !loop over ia      
        dA=Covs(ia,ia)             !extract da   
        Corrs(ia,ib)=Covs(ia,ib)/sqrt(dA*dB)
       end do                      !close ia-loop
      end do                       !close ib-loop
c       
      return
      end
c-----------------------------------------------------------------------
      subroutine Writer
c      
      implicit real*8 (a-h,j-z) 
      parameter (iNmax=10000)
      common/Configs/X0(8,iNmax)
      common/X0Covariances/Covs(8,8),Corrs(8,8)  
      dimension A(iNmax),AvgA(8),dA(8)
c     
c     Write MLE parameters w/errors to file:
      do ia=1,8                    !loop over ia
       A(:)=X0(ia,:)               !extract A 
       call Avg(A,AvgA(ia),iNmax)  !<A>    
       dA(ia)=sqrt(Covs(ia,ia))    !standard deviation
      end do                       !close ia-loop 
      write(9,100) (AvgA(i),dA(i),i=1,8)
c     
c     Write Correlation coefficients:      
      write(9,200)              !write to file
      do ia=1,8                 !loop over ia
       write(9,300) Corrs(ia,:) !write to file
      end do      
c
c     Write all Configurations:
      write(9,400)              !write to file
      do in=1,iNmax             !loop over configurations
       write(9,500) in,(X0(im,in),im=1,8)  !write to file 
      end do                    !close do      
c
 100  format('#'                      ,/,
     .     '#',3x,' Averages and Uncertainties:',/,
     .     '#',3x,' BE=',f12.8,' +/-',f12.8,/,
     .     '#',3x,'Rho=',f12.8,' +/-',f12.8,/,
     .     '#',3x,' M*=',f12.8,' +/-',f12.8,/,
     .     '#',3x,' K =',f12.8,' +/-',f12.8,/,
     .     '#',3x,' J =',f12.8,' +/-',f12.8,/,
     .     '#',3x,' L =',f12.8,' +/-',f12.8,/,
     .     '#',3x,' Zt=',f12.8,' +/-',f12.8,/,
     .     '#',3x,' Ms=',f12.8,' +/-',f12.8) 
 200  format('#'                      ,/,
     .     '#',3x,' Correlation Coefficients:',/,
     .     '#',14x,'BE',14x,'Rho',15x,'M*',15x,'K',
     .         16x,'J',16x,'L',14x,'Zeta',15x,'Ms')
 300  format('#',3x,8(' ',f16.9))
 400  format('#'                      ,/,
     .     '#',3x,'Configurations as per Covariance Matrix:',/,
     .     '#',14x,'BE',14x,'Rho',15x,'M*',15x,'K',
     .         16x,'J',16x,'L',14x,'Zeta',15x,'Ms')
 500  format(i6,1x,8('',d17.9))
c      
      return
      end
c-----------------------------------------------------------------------
c-----------------------------------------------------------------------
c     Compute the covariance between A and B.
c     
      subroutine Covar(A,B,Cab,iNmax)
c
      implicit real*8 (a-h,j-z)
      dimension A(10000),B(10000)
c
c     Average of A*B:
      AvgAB=0.0000d0                    !initialize <A*B>
      do in=1,iNmax                     !loop over values
       AvgAB=Avgab+A(in)*B(in)          !update <A*B> 
      end do                            !close in-loop
      AvgAB=AvgAB/iNmax                 !<A*B>      
c
c     Covariance of A and B:
      call Avg(A,AvgA,iNmax)            !<A>
      call Avg(B,AvgB,iNmax)            !<B>
      Cab=AvgAB-AvgA*AvgB               !cov(AB)=<A*B>-<A>*<B>
      
      return
      end
c-----------------------------------------------------------------------
c     Compute the average of A.
c     
      subroutine Avg(A,AvgA,iNmax)
c
      implicit real*8 (a-h,j-z)
      dimension A(10000)
c
c     Average of A:
      AvgA=0.0000d0                     !initialize <A>
      do in=1,iNmax                     !loop over values
       AvgA=AvgA+A(in)                  !update <A>
      end do                            !close in-loop
      AvgA=AvgA/iNmax                   !<A> 
c     
      return
      end
c-----------------------------------------------------------------------      
