import numpy as np

def sym(X) : 
    return((X+X.T)/2)

def stand(x) : 
    dia = np.apply_along_axis(np.var,0,x)
    dia[dia!=0] = dia[dia!=0]**(-0.5)
    return(np.matmul(x-np.apply_along_axis(np.mean,0,x),np.diag(dia)))

def ridgepower(mat,power,eps):
    big = np.linalg.eig(mat)[0][0] 
    return(matpower(mat+np.diag(np.repeat(eps*big,mat.shape[0])),power))
    
def matpower(mat,power,thre = 0.00000001):
    mat = (mat + mat.T)/2
    eig = np.linalg.eig(mat)
    val = np.diag([np.real(i**power) if i >thre else 0 for i in eig[0]])
    vec = np.real(eig[1])
    return(np.matmul(np.matmul(vec,val),vec.T))


def PCA(X):
    eig = np.linalg.eig(np.matmul(X.T,X))
    return([eig[1],eig[0]])

def SIR(y,X,h):
    n = X.shape[0];p = X.shape[1] 
    signrt = matpower(np.diag(np.var(X)),-0.5)
    X = stand(X)
    ylabel=[]
    ylabel = np.unique(np.array(y))
    prob = np.array([]);exy =np.array([np.repeat(0,p)])
    for i in range(h):
        prob = np.append(prob, y[np.array(y==ylabel[i])].shape[0]/n)
    for i in range(h):
        exy = np.concatenate([exy,[np.apply_along_axis(np.mean,0,X.iloc[np.array(y==ylabel[i]),:])]],axis=0)
    exy = exy[1:,:]
    sirmat = np.matmul(np.matmul(exy.T,np.diag(prob)),exy)
    eig = np.linalg.eig(sirmat)
    return([np.matmul(signrt,eig[1]),eig[0]])

def DR(y,X,h) :
    n = X.shape[0];p = X.shape[1] 
    signrt = matpower(np.diag(np.var(X)),-0.5)
    X = stand(X)
    yless = np.array(y);ylabel=[]
    ylabel = np.unique(np.array(y))
    prob = np.array([]);exy =np.array([np.repeat(0,p)])
    for i in range(h):
        prob = np.append(prob, y[np.array(y==ylabel[i])].shape[0]/n)
    vxy = np.zeros(shape=(p,p,h))
    for i in range(h):
        vxy[:,:,i] = np.cov(X.iloc[np.array(y==ylabel[i]),].T)
        exy = np.concatenate([exy,[np.apply_along_axis(np.mean,0,X.iloc[np.array(y==ylabel[i]),:])]],axis=0)
    exy = exy[1:,:]
    mat1 = np.zeros(shape=(p,p));mat2 = np.zeros(shape=(p,p))
    for i in range(h):
        xxt = np.matmul(exy[i:i+1,:].T,exy[i:i+1,:])
        mat1 = mat1+prob[i]*np.matmul((vxy[:,:,i]+xxt),vxy[:,:,i]+xxt)
        mat2 = mat2+prob[i]*xxt
    out = 2*mat1+2*np.matmul(mat2,mat2)+2*np.sum(np.diag(mat2))*mat2-2*np.identity(p)
    eig = np.linalg.eig(out)
    return([np.real(np.matmul(signrt,eig[1])),eig[0]])
