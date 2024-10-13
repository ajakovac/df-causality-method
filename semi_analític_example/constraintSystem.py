import numpy as np
import time

class status:
    def __init__(self, imax, rateUnit = 1):
        self.unit = rateUnit
        self.imax = imax
        self.initTime = time.time()
        self.prevRate = 0
        self.elapsedTime = 0
        self.print(0)

    def __call__(self, i, message=""):
        rate = 100*(i+1)/self.imax
        if int(rate/self.unit)- self.prevRate >= 1:
            self.prevRate = int(rate/self.unit)
            self.print(rate, message)
    
    def print(self, rate, message=""):
        self.elapsedTime = time.time()-self.initTime
        toPrint = '--> '
        toPrint += f'status={rate:5.1f}%'
        toPrint += f', (dt={self.elapsedTime:.1f}sec'
        if rate > 0:
            expTime = 100*self.elapsedTime/rate
            toPrint += f', tmax={expTime:.1f}sec, tleft={expTime-self.elapsedTime:.1f}sec'
        toPrint += ')' 
        if message:
            toPrint += f', message = {message}'
        print(toPrint, end='\r', flush=True)

class constraintSystem :
    def __init__(self, Q, S, Niter=10000, kmax=10):
        self.N = Q.shape[0]
        self.kmax = kmax
        self.Q = Q

        if len(np.array(S).shape) == 0:
            self.sigma2 = S**2*np.eye(self.N)
        else:
            self.sigma2 = S@S.T
            if self.sigma2.shape[0] != self.N:
                raise ValueError("size mismatch")

        Clist = [self.sigma2]
        St = status(Niter)
        for ni in range(Niter):
            St(ni)
            Clist.append(Q@Clist[-1]@Q.T)
        Clist=np.array(Clist)
        self.C = np.sum(Clist, axis=0)

        self.H = np.linalg.inv(Q)
        self.Czeta = np.zeros(shape = (self.kmax-1,self.kmax-1,self.N,self.N))
        for k in range(1, self.kmax):
            for ell in range(1,self.kmax):
                for m in range(0, min(k,ell)):
                    Hkl = self.H**(k-m) @ self.sigma2 @ (self.H.T)**(ell-m)
                    self.Czeta[k-1,ell-1] = Hkl
    
    def system(self, coords, sigmaW2, zvector = None):
        Nconstr = len(coords)

        if len(np.array(sigmaW2).shape) == 0:
            CK = sigmaW2*np.eye(Nconstr)
        else:
            if len(sigmaW2) != Nconstr:
                raise ValueError("size mismatch")
            CK = np.diag(sigmaW2)
        
        if zvector is None:
            zvector = np.zeros(shape=(Nconstr,))

        R = np.zeros(shape=(Nconstr,self.N))
        for aa in range(Nconstr):
            k,a = coords[aa]
            U = self.H**k
            R[aa] = U[a,:]

        
        for aa in range(Nconstr):
            k,a = coords[aa]
            CK[aa,aa] += self.Czeta[k-1,k-1,a,a]
            for bb in range(aa-1):
                ell,b = coords[bb]
                CK[aa,bb] = self.Czeta[k-1,ell-1,a,b]
                CK[bb,aa] = self.Czeta[k-1,ell-1,a,b]

        CKinv = np.linalg.inv(CK)
        Cx = np.linalg.inv(np.linalg.inv(self.C) + R.T@CKinv@R)

        return Cx, Cx@R.T@CKinv@zvector

