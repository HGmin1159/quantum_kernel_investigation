import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from tqdm import tqdm


class Kernel_Regression():
    def __init__(self,thre = 10**(-8)) :
        self.thre = thre
    
    def fit(self,Y_train):
        self.Y_train = Y_train

    def predict(self,G_test):
        return(np.matmul(G_test,self.Y_train)/np.apply_along_axis(np.sum,1,np.asarray(G_test)))


class SVM():
    def __init__(self,lamda=1):
        self.lamda = lamda
    
    def fit(self,kernel_matrix,y_train_ori):

        y_train = np.asarray(2.0*(y_train_ori==list(set(y_train_ori))[0])-1).reshape(-1)
        k=1000

        n = len(kernel_matrix[0])
        upbound = 2*n*self.lamda
        Q_mat = np.matmul(np.matmul(np.diag(y_train),kernel_matrix),np.diag(y_train))
        Q = matrix(Q_mat)
        r = matrix(k*np.array([1.0 for i in range(n)]))

        G = matrix(np.concatenate([-np.eye(n),np.eye(n)]))
        h = matrix([0.0 for i in range(n)] + [upbound**(-1) for i in range(n)])
        A = matrix(y_train).T
        b = matrix(0.0)

        sol = qp(Q,r,A,b,G,h, kktsolver='ldl', options={'kktreg':1e-9,'show_progress': False})

        c_sol = [sol["x"]]
        self.coef = [i for i in c_sol[0]]
        self.train_gram = kernel_matrix
        self.y_train = y_train
        self.y_train_ori = y_train_ori
    
    def predict(self,kernel_matrix_pred):
        y_pred = np.sign(np.matmul((self.coef * self.y_train),kernel_matrix_pred))
        return(y_pred)
    
    def score(self,kernel_matrix_pred,y_true):
        n = len(y_true)
        y_true = np.asarray(2.0*(y_true==list(set(self.y_train_ori))[0])-1).reshape(-1)
        y_pred = self.predict(kernel_matrix_pred)
        return(sum(y_pred==y_true)/n)


class KPCA():
    def __init__(self,thre = 10**(-8)):
        self.thre = thre
     
    def fit(self,kernel_matrix) :
        n = len(kernel_matrix)
        kernel_matrix = np.array(kernel_matrix)
        eig = np.linalg.eigh(kernel_matrix)
        invorder = [n-i-1 for i in range(n)]
        V = eig[1][:,invorder]
        U = np.matmul(matpower(kernel_matrix,-0.5,self.thre),V)
        self.candi_matrix = kernel_matrix
        self.coef = np.array(U)
        self.train_gram = kernel_matrix
        self.eigvalue = eig[0]
    
    def transform(self,kernel_matrix_pred) :
        return(np.matmul(np.array(kernel_matrix_pred).T,self.coef))

class GSIR():
    def __init__(self,lamda_x = 0.001,lamda_y = 0.001, y_type="Discrete",a_type = "Inverse",comp=3):
        self.lamda_x = lamda_x
        self.lamda_y = lamda_y
        self.type = y_type
        self.comp = comp
        self.a_type = a_type

    def fit(self,kernel_matrix,y_train):
        n = len(kernel_matrix)
        Q = np.identity(n)-np.ones((n,n))/n
        if self.type == "Discrete" : 
            G_y = Gram_discrete(y_train)
            G_yinv = np.linalg.pinv(sym(G_y))
        elif self.type == "Continuous" : 
            G_y = Gram_gaussian(y_train,comp=self.comp)
            G_yinv = np.linalg.pinv(sym(G_y+self.lamda_y * np.identity(n)))
        G_X = kernel_matrix
        Ginv = np.linalg.pinv(sym(G_X+self.lamda_x*np.identity(n)))
        
        if self.a_type == "Identity" : candi_matrix = np.matmul(np.matmul(np.matmul(np.matmul(Ginv,G_X),G_y),G_X),Ginv)
        elif self.a_type == "Inverse" : candi_matrix = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(Ginv,G_X),G_y),G_yinv),G_X),Ginv)
        eig = np.linalg.eigh(candi_matrix)
        invorder = [n-i-1 for i in range(n)]
        V = eig[1][:,invorder]
        self.candi_matrix = candi_matrix
        self.coef = np.matmul(np.matmul(Q,Ginv),V)
        self.train_gram = kernel_matrix
        self.eigvalue = eig[0]

    def transform(self,kernel_matrix_pred):
        return(np.matmul(np.array(kernel_matrix_pred).T,self.coef))