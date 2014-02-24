subroutine unknownArg(wrap, a, b, i1, i2, n, nLocal, numDeps, coupling, rands, &
     indices, cplFactors)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) wrap, a, b, i1, i2, n, nLocal, numDeps, coupling, rands
  !f2py intent(out) indices, cplFactors
  !f2py depend(numDeps) rands
  !f2py depend(nLocal, numDeps) indices, cplFactors

  !Input
  logical, intent(in) ::  wrap
  integer, intent(in) ::  a, b, i1, i2, n, nLocal, numDeps
  double precision, intent(in) ::  coupling, rands(numDeps)

  !Output
  integer, intent(out) ::  indices(i1:i2-1, numDeps)
  double precision, intent(out) ::  cplFactors(i1:i2-1, numDeps)

  !Working
  integer i, j, jj

  do i=i1,i2
     indices(i,1) = i
     cplFactors(i,1) = 0.0
     do jj=2,numDeps
        j = i + a + nint((b-a) * rands(jj))
        if ((0 .le. j) .and. (j .lt. n)) then
           indices(i,jj) = j
           cplFactors(i,jj) = exp(-coupling * abs(i-j))
        else if (wrap .and. (j .lt. 0)) then
           indices(i,jj) = j + n
           cplFactors(i,jj) = exp(-coupling * abs(i-j))
        else if (wrap .and. (j .ge. n)) then
           indices(i,jj) = j - n
           cplFactors(i,jj) = exp(-coupling * abs(i-j))
        else
           indices(i,jj) = i
           cplFactors(i,jj) = 0.0
        end if
     end do
  end do

end subroutine unknownArg




subroutine parameterArg(n, i1, i2, nLocal, nArg, numDeps, indices)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) n, i1, i2, nLocal, nArg, numDeps
  !f2py intent(out) indices
  !f2py depend(nLocal, numDeps) indices

  !Input
  integer, intent(in) ::  n, i1, i2, nLocal, nArg, numDeps

  !Output
  integer, intent(out) ::  indices(i1:i2-1, numDeps)

  !Working
  integer i, j

  if (nArg .le. numDeps) then
     do j=1,numDeps
        indices(:,j) = nint(1.0*j/(numDeps-1) * (nArg-1))
     end do
  else
     do i=i1,i2
        do j=1,numDeps
           indices(i,j) = nint(1.0*i/(n-1) * (nArg-numDeps)) + j-1
        end do
     end do
  end if

end subroutine parameterArg




subroutine evalC(n, numArgs, i1, i2, degree, &
     scaling, condNum, nonlin, cplFactors, v, C)

  implicit none

  !Fortran-python interface directives
  !f2py intent(in) n, numArgs, i1, i2, degree, scaling, condNum, nonlin, cplFactors, v
  !f2py intent(out) C
  !f2py depend(n, numArgs) cplFactors
  !f2py depend(n) v, C

  !Input
  integer, intent(in) ::  n, numArgs, i1, i2, degree
  double precision, intent(in) ::  scaling, condNum, nonlin
  double precision, intent(in) ::  cplFactors(i1:i2-1, numArgs), v(numArgs)

  !Output
  double precision, intent(out) ::  C(i1:i2-1)

  !Working
  integer i, j, k, r, s, factorial
  integer, allocatable, dimension(:) ::  jj
  double precision res1, res2, res3

  do i=i1,i2
     res3 = 0
     do r=1,degree
        res2 = 0
        allocate(jj(r))
        jj(:) = 1
        do k=1,numArgs*r
           res1 = 1
           do s=1,r
              j = jj(s)
              res1 = res1 * v(j) * cplFactors(i,j)
           end do
           call increment(numArgs, r, jj)
           res2 = res2 + res1
        end do
        deallocate(jj)
        res2 = res2 / factorial(r) * exp(-nonlin * r)
        res3 = res3 + res2
     end do
     res3 = res3 * scaling * 10**(-condNum/2.0 + condNum*(i-1)/(n-1))
     C(i) = res3
  end do

end subroutine evalC



subroutine increment(m, n, inds)

  implicit none

  !Input
  integer, intent(in) ::  m, n

  !Output
  integer, intent(inout) ::  inds(n)

  !Working
  integer i

  loop: do i=1,n
     if (inds(i) .lt. m) then
        inds(i) = inds(i) + 1
        exit loop
     end if
  end do loop

end subroutine increment



function factorial(n)

  implicit none

  integer, intent(in) ::  n
  integer factorial, i
  
  factorial = 1
  do i=1,n
     factorial = factorial * i
  end do

end function factorial
