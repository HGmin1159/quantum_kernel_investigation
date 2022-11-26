#################################################################
# Index
# 0. Import Base Packages 
# 1. Functions for toy experiment about kernel
# 2. Functions for Quantum Kernel Estimation
# 3. Functions for Nonlinear Dimension Reduction
# 4. Functions for Kernel
##################################################################

##################################################################
# 0. Import Base Packages 
##################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit , transpile
from qiskit.visualization import *
from qiskit.tools.jupyter import *
from qiskit.providers.aer import QasmSimulator

from qiskit import  QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms.optimizers import *
from qiskit_machine_learning.circuit.library import RawFeatureVector
from tqdm import tqdm

##################################################################
# 1. Function for toy experiment about kernel
##################################################################

def amplitude_encoding(num_qubits,x):
    if 2**num_qubits < len(x) : return("Need More Qubits") 
    data = np.zeros(2**num_qubits)
    data[:len(x)]=x
    qc = RawFeatureVector(num_qubits,name ="Amplitude Encoding")
    qc.assign_parameters(data)
    encode = qc.to_gate()
    return encode

def qubit_encoding(num_qubits,x):
    qc = QuantumCircuit(num_qubits,name = "Qubit Encoding")
    for i in range(num_qubits):
        qc.ry(x[i], [i], label='qubit_encoding X'+str(i))
    encode = qc.to_gate()
    return encode

def swap(x_1,x_2,kernel,n_kernel):
    qc = QuantumCircuit(n_kernel*2+1,1)
    qc.h(n_kernel*2)

    x_1 = kernel(n_kernel,x_1)
    x_2 = kernel(n_kernel,x_2)
    qc.append(x_1,range(n_kernel))
    qc.append(x_2,range(n_kernel,(n_kernel*2)))

    for i in range(n_kernel):
        qc.cswap(n_kernel*2,i,i+n_kernel)
    qc.h(n_kernel*2)
    qc.measure(n_kernel*2,0)
    return(qc)
    
def hadamard(x_1,x_2,kernel,n_kernel):
    qc = QuantumCircuit(n_kernel+1,1)
    qc.h(n_kernel)

    x_1 = kernel(n_kernel,x_1).control(1)
    x_2 = kernel(n_kernel,x_2).control(1)
    qc.append(x_1,[n_kernel,*[i for i in range(n_kernel)]])
    qc.x(n_kernel)
    qc.append(x_2,[n_kernel,*[i for i in range(n_kernel)]])
    qc.h(n_kernel)
    qc.measure(n_kernel,0)
    return(qc)

def hadamard_inner(x1,x2,kernel,n_kernel,shots=10000):
    hadamard_circuit = hadamard(x1,x2,kernel,n_kernel)
    
    backend = QasmSimulator()
    qc_compiled = transpile(hadamard_circuit, backend)
    
    job = backend.run(qc_compiled, shots=shots)
    counts = job.result().get_counts ()
    
    hatpr0 = counts["0"]/shots
    quantum_inner = 2*hatpr0-1
    return(quantum_inner)

def swap_inner(x1,x2,kernel,n_kernel,shots=10000):
    circuit = swap(x1,x2,kernel,n_kernel)
    
    backend = QasmSimulator()
    qc_compiled = transpile(circuit, backend)
    
    job = backend.run(qc_compiled, shots=shots)
    counts = job.result().get_counts ()
    
    hatpr0 = counts["0"]/shots
    quantum_inner = np.sqrt(2*hatpr0-1)
    return(quantum_inner)

def swap_test(x_1,x_2,kernel):
    
    First_kernel = kernel(x_1)
    Second_kernel = kernel(x_2)
    n_kernel = First_kernel[1]
    
    qc = QuantumCircuit(n_kernel*2+1,1)
    qc.h(n_kernel*2)

    qc.append(First_kernel[0],range(n_kernel))
    qc.append(Second_kernel[0],range(n_kernel,(n_kernel*2)))

    for i in range(n_kernel):
        qc.cswap(n_kernel*2,i,i+n_kernel)
    qc.h(n_kernel*2)
    qc.measure(n_kernel*2,0)
    return(qc)

def hadamard_test(x_1,x_2,kernel):
    First_kernel = kernel(x_1)
    Second_kernel = kernel(x_2)
    n_kernel = First_kernel[1]
    
    qc = QuantumCircuit(n_kernel+1,1)
    qc.h(n_kernel)

    x_1 = First_kernel[0].control(1)
    x_2 = Second_kernel[0].control(1)
    qc.append(x_1,[n_kernel,*[i for i in range(n_kernel)]])
    qc.x(n_kernel)
    qc.append(x_2,[n_kernel,*[i for i in range(n_kernel)]])
    qc.h(n_kernel)
    qc.measure(n_kernel,0)
    return(qc)

def hadamard_test_img(x_1,x_2,kernel):
    First_kernel = kernel(x_1)
    Second_kernel = kernel(x_2)
    n_kernel = First_kernel[1]
    
    qc = QuantumCircuit(n_kernel+1,1)
    qc.h(n_kernel)
    qc.s(n_kernel)
    
    x_1 = First_kernel[0].control(1)
    x_2 = Second_kernel[0].control(1)
    qc.append(x_1,[n_kernel,*[i for i in range(n_kernel)]])
    qc.x(n_kernel)
    qc.append(x_2,[n_kernel,*[i for i in range(n_kernel)]])
    qc.x(n_kernel)
    qc.h(n_kernel)
    qc.measure(n_kernel,0)
    return(qc)


##################################################################
# 2. Function for Quantum Kernel Estimation
##################################################################

def qke(x1,x2,kernel,layer=5, backend=QasmSimulator(),shots=1000):
    gate1,n_qubits = kernel(x1,layer)
    gate2 = QuantumCircuit.inverse(kernel(x2,layer)[0])
    
    qc = QuantumCircuit(n_qubits)
    qc.append(gate1.to_gate(),range(n_qubits))
    qc.append(gate2.to_gate(),range(n_qubits))
    qc.measure_all()
    
    qc_compiled = transpile(qc, backend)
    job = backend.run(qc_compiled, shots=shots)
    counts = job.result().get_counts ()
    
    string = ""
    for k in range(n_qubits) : string += "0"
    if (string in counts.keys()): return(counts[string]/shots)
    else : return(0)

def qke_multi(x1,x2,kernel,layer=5, backend=QasmSimulator(),shots=1000):
    n = len(x1)
    k = n//32
    layer = 1
    qc = QuantumCircuit(8,8)
    for i in range(k):
        gate1,n_qubits = kernel(x1[i*32:(i+1)*32],layer)
        qc.append(gate1.to_gate(),range(8))
    gate1,n_qubits = kernel(x1[k*32:n],layer)
    qc.append(gate1.to_gate(),range(n_qubits))

    for i in range(k):
        gate2 = QuantumCircuit.inverse(kernel(x2[i*32:(i+1)*32],layer)[0])
        qc.append(gate2.to_gate(),range(8))
    gate2 = QuantumCircuit.inverse(kernel(x2[k*32:n],layer)[0])
    qc.append(gate1.to_gate(),range(n_qubits))
    
    qc.measure_all()

    qc_compiled = transpile(qc, backend)
    job = backend.run(qc_compiled, shots=shots)
    counts = job.result().get_counts ()
    
    string = ""
    for k in range(n_qubits) : string += "0"
    if (string in counts.keys()): return(counts[string]/shots)
    else : return(0)

def quantum_inner(method,x1,x2,kernel,shots=10000):
    if method == "hadamard" : test = hadamard_test
    elif method == "swap" : test = swap_test
    else : raise NameError('Wrong method name')
    
    circuit = test(x1,x2,kernel)
    
    backend = QasmSimulator()
    qc_compiled = transpile(circuit, backend)
    
    job = backend.run(qc_compiled, shots=shots)
    counts = job.result().get_counts ()
    
    hatpr0 = counts["0"]/shots
    if method == "hadamard" : quantum_inner = 2*hatpr0-1
    elif method == "swap" : quantum_inner = np.sqrt(2*hatpr0-1)
    return(quantum_inner)

def classic_qubit_encoding(theta) :
    result = [1]
    for i in theta :
        result_temp = []
        for j in result :
            result_temp = [*result_temp,*[j*np.cos(i),j*np.sin(i)]]
        result = result_temp
    return(result)

def kernel_test(x):
    qc = QuantumCircuit(1,name = "Kernel test")
    qc.rx(x[0],0)
    encode = qc.to_gate()
    return [encode,1]

def kernel_circuit(kernel,data):
    
    if type(data) == int : 
        theta = []
        for i in range(data):
            theta = theta+[Parameter("theta"+str(i))]
    else : theta = data
        
    gate,num_qubit = kernel(theta)
    qc = QuantumCircuit(num_qubit)
    qc.append(gate,range(num_qubit))
    return(qc)

def get_gram(data,kernel_fun,layer,backend = QasmSimulator(),shots=1000):
    n = len(data)
    gram_matrix = np.identity(n)
    for i in tqdm(range(n)):
        for j in range(i):
            gram_matrix[i,j] = qke(data.iloc[i,:].tolist(),data.iloc[j,:].tolist(),kernel_fun,layer,backend,shots=shots)
            gram_matrix[j,i] = gram_matrix[i,j]
    return(gram_matrix)

def get_gram_test(data,test_data,kernel_fun,layer,backend = QasmSimulator(),shots=1000):
    n = len(data)
    m = len(test_data)
    gram_matrix = np.empty(shape=(n,m))
    for i in tqdm(range(n)):
        for j in range(m):
            gram_matrix[i,j] = qke(data.iloc[i,:].tolist(),test_data.iloc[j,:].tolist(),kernel_fun,layer,backend,shots=shots)
    return(gram_matrix)

def get_gram_multi(data,kernel_fun,layer,backend = QasmSimulator(),shots=1000):
    n = len(data)
    gram_matrix = np.identity(n)
    for i in tqdm(range(n)):
        for j in range(i):
            gram_matrix[i,j] = qke_multi(data.iloc[i,:].tolist(),data.iloc[j,:].tolist(),kernel_fun,layer,backend,shots=shots)
            gram_matrix[j,i] = gram_matrix[i,j]
    return(gram_matrix)

def get_gram_test_multi(data,test_data,kernel_fun,layer,backend = QasmSimulator(),shots=1000):
    n = len(data)
    m = len(test_data)
    gram_matrix = np.empty(shape=(n,m))
    for i in tqdm(range(n)):
        for j in range(m):
            gram_matrix[i,j] = qke_multi(data.iloc[i,:].tolist(),test_data.iloc[j,:].tolist(),kernel_fun,layer,backend,shots=shots)
    return(gram_matrix)



##################################################################
# 3. Function for Nonlinear Dimension Reduction
##################################################################

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

def Gram_discrete(data,version="H") :  
    n = data.shape[0]
    J = np.outer(np.ones(shape=(n,1)),np.ones(shape=(n,1)))
    Q = np.identity(n)-J/n
    K = [[1 if i==j else 0 for j in data] for i in data]
    if(version == "H"):
        dis = np.matmul(np.matmul(Q,K),Q)
    if(version == "L"):
        dis = np.concatenate([np.ones((1,n)),K])
    return(dis)

def Gram_gaussian(data,comp,version="H") :
    n = data.shape[0];p = data.shape[1] 
    U = np.matmul(data,data.T)
    M = np.outer(np.diag(U),np.ones(shape=(n,1)))
    J = np.outer(np.ones(shape=(n,1)),np.ones(shape=(n,1)))
    Q = np.identity(n)-J/n
    K = M+M.T-2*U
    sigma = (np.sum(np.sqrt(K))-np.trace(np.sqrt(K)))/n/(n-1)
    gamma = comp/sigma/sigma
    result = np.exp(-K*gamma)
    return(result)

def Gram_gaussian_test(data1,data2,comp,version="H") :
    n = data1.shape[0];p = data1.shape[1] 
    m = data2.shape[0]

    U = np.matmul(data1,data1.T)
    M = np.outer(np.diag(U),np.ones(shape=(n,1)))
    K = M+M.T-2*U

    M_new = np.outer(np.diag(U),np.ones(shape=(m,1)))
    U_new = np.matmul(data1,data2.T)
    T_new = np.outer(np.diag(np.matmul(data2,data2.T)),np.ones(shape=(n,1)))
    K_new = M_new+T_new.T-2*U_new

    sigma = (np.sum(np.sqrt(K))-np.trace(np.sqrt(K)))/n/(n-1)
    gamma = comp/sigma/sigma
    result = np.exp(-K_new*gamma)
    return(result)

def KPCA(Gram_X,thre = 10**(-8)) :
    G = Gram_X
    eig = np.linalg.eig(G)
    V = np.real(eig[1])
    U = np.matmul(matpower(G,-0.5,10**(-8)),V)
    return([np.matmul(G,U),eig[0]])

def GSIR(Gram_y,Gram_X):
    G_y = Gram_y
    G_X = Gram_X
    Ginv = np.linalg.pinv(G_X)
    candi_matrix = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(Ginv,G_X),G_y),Ginv),G_X),Ginv)
    eig = np.linalg.eig(candi_matrix)
    V = np.real(eig[1])
    eig_score = eig[0]
    return([np.matmul(np.matmul(G_X,Ginv),V),eig_score])

def GSAVE(Gram_y,Gram_X):
    n = Gram_X.shape[0]
    L_y = Gram_y
    L_X = Gram_X
    J = np.outer(np.ones(shape=(n,1)),np.ones(shape=(n,1)))
    Q = np.identity(n)-J/n
    lql_X = np.matmul(L_X,np.matmul(Q,L_X.T))
    lql_X_inv = np.matmul(ridgepower(lql_X,-0.5,0.000001),np.matmul(L_X,Q))
    
    tau = np.matmul(np.matmul(L_y.T,ridgepower(np.matmul(L_y,L_y.T),-1,0.000001)),L_y)
    a0 = np.diag(np.apply_along_axis(sum,0,tau)) - np.matmul(tau,tau)
    a1 = np.diag(np.apply_along_axis(sum,0,tau*tau))-np.matmul(tau,tau)/n
    a2 = np.matmul((tau*tau),tau)-np.matmul(np.matmul(tau,np.diag(np.apply_along_axis(np.mean,0,tau))),tau)
    a3 = np.matmul(np.matmul(tau,np.diag(np.diag(np.matmul(tau,np.matmul(Q,tau))))),tau)
    mid = Q/n -(2/n)*np.matmul(np.matmul(Q,a0),Q) +np.matmul(np.matmul(Q,(a1-a2-a2.T+a3)),Q)
    candi_matrix = np.matmul(lql_X_inv,np.matmul(mid,lql_X_inv.T))
    eig = np.linalg.eig(candi_matrix)
    eigp = pd.DataFrame(eig[1].T)
    eig_score = eig[0]
    eigp["value"]  = eig[0]
    V = eigp.sort_values("value",ascending=False).iloc[:,:(n+1)].values.T
    return([np.matmul(lql_X_inv.T,V),eig_score])

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

def kernel_density(G_kde,index):
    kde= pd.DataFrame()
    kde["index"] = index
    kde["prob"] = pd.DataFrame(G_kde).apply(sum)/sum(pd.DataFrame(G_kde).apply(sum))
    kde = kde.sort_values("index")
    plt.plot(kde.iloc[:,0],kde.iloc[:,1])
    
def kernel_regression(G_kr,y,index,sto_index,label=None):
    kr = pd.DataFrame()
    kr["index"] = index
    kr["estimates"] = np.matmul(G_kr,y[sto_index])/sum(G_kr)
    kr = kr.sort_values("index")
    plt.plot(kr.iloc[:,0],kr.iloc[:,1],label=label)

def kernel_regression_MSE(G_kr,y,index,sto_index):
    size = len(y)
    kr = pd.DataFrame()
    kr["index"] = index
    kr["estimates"] = np.matmul(G_kr,y[sto_index])/sum(G_kr)
    kr = kr.sort_values("index")
    y_pred = interpolate(kr).iloc[:,1]
    result = sum((y_pred-y)**2)/size
    return(result)


##################################################################
# 4. Function for Kernel
##################################################################

def zz_kernel(x,repeat=1):
    p = len(x)
    qc = ZZFeatureMap(p, reps=repeat)
    qc = qc.bind_parameters({qc.parameters[i]:x[i] for i in range(p)})
    encode = qc
    return [encode,p]

def exponential_kernel_B(x,repeat=5):
    qc = QuantumCircuit(repeat)
    for i in range(repeat) :
        qc.rx(x[0]*3**i,[i])
    for i in range(repeat-1) :
        qc.cx([i],[i+1])
    qc.cx([repeat-1],[0])
    for i in range(repeat) :
        qc.ry(x[1]*3**i,[i])
    for i in range(repeat-1) :
        qc.cx([i],[i+1])
    qc.cx([repeat-1],[0])
    encode = qc
    return [encode,repeat]

def simple_kernel_A(x,repeat=1):
    qc = QuantumCircuit(repeat)
    for i in range(repeat) :
        qc.rx(x[0],[i])
    for i in range(repeat-1) :
        qc.cx([i],[i+1])
    qc.cx([repeat-1],[0])
    encode = qc
    return [encode,repeat]

def kernel_A(x,repeat=1):
    r = 2
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel A")
    for i in range(num_qubits):
        qc.rx(x[i], [i])
    for i in range(num_qubits):
        qc.rz(x[i+num_qubits], [i])
    encode = qc
    return [encode,num_qubits]

def kernel_B(x,repeat=1):
    r = 2
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel B")
    for i in range(num_qubits):
        qc.rx(x[i], [i])
    for i in range(num_qubits):
        qc.rz(x[i+num_qubits], [i])
    for i in range(num_qubits-1):
        qc.cx([i+1],[i])
    qc.cx([0],[num_qubits-1])
    encode = qc
    return [encode,num_qubits]

def kernel_C(x,repeat=1):
    r = 3
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel C")
    for i in range(num_qubits):
        qc.rx(x[i], [i])
    for i in range(num_qubits):
        qc.rz(x[i+num_qubits], [i])
    for i in range(num_qubits-1):
        qc.crz(x[i+2*num_qubits],[i+1],[i])
    qc.crz(x[3*num_qubits-1],[0],[num_qubits-1])
    encode = qc
    return [encode,num_qubits]

def kernel_D(x,repeat=1):
    r = 3
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel D")
    for i in range(num_qubits):
        qc.rx(x[i], [i])
    for i in range(num_qubits):
        qc.rz(x[i+num_qubits], [i])
    for i in range(num_qubits-1):
        qc.crz(x[i+2*num_qubits],[i+1],[i])
    qc.crx(x[3*num_qubits-1],[0],[num_qubits-1])
    encode = qc
    return [encode,num_qubits]

def kernel_E(x,repeat=1):
    r = 1
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel E")
    for i in range(num_qubits):
        qc.h([i])
    for i in range(num_qubits-1):
        qc.cz([i],[i+1])
    qc.cz(num_qubits-1,0)
    for i in range(num_qubits):
        qc.rz(x[i],[i])
    encode = qc
    return [encode,num_qubits]

def kernel_F(x,repeat=1):
    r = 2
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel F")
    for i in range(num_qubits):
        qc.ry(x[i],[i])
    for i in range(num_qubits-1):
        qc.cz([i],[i+1])
    qc.cz(num_qubits-1,0)
    for i in range(num_qubits):
        qc.ry(x[i+num_qubits],[i])
    encode = qc
    return [encode,num_qubits]

def kernel_G(x,repeat=1):
    r = 2
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel G")
    for i in range(num_qubits):
        qc.ry(x[i],[i])
    for i in range(num_qubits-1):
        qc.cx([i+1],[i])
    qc.cx(0,num_qubits-1)

    for i in range(num_qubits):
        qc.ry(x[i+num_qubits],[i])
    for i in range(num_qubits-1):
        qc.cx([i],[i+1])
    qc.cx(num_qubits-1,0)
    encode = qc
    return [encode,num_qubits]

def kernel_H(x,repeat=1):
    r = 4
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel H")
    for i in range(num_qubits):
        qc.ry(x[i],[i])
    for i in range(num_qubits):
        qc.rz(x[num_qubits+i],[i])
    former = int((num_qubits) /2)
    later = int((num_qubits-1) /2)
    for i in range(former):
        qc.cx([2*i+1],[2*i])
    for i in range(num_qubits):
        qc.ry(x[2*num_qubits + i],[i])
    for i in range(num_qubits):
        qc.rz(x[3*num_qubits+i],[i])
    for i in range(later):
        qc.cx([2*i+2],[2*i+1])
    encode = qc
    return [encode,num_qubits]

def kernel_I(x,repeat=1):
    r = 4
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel I")
    for i in range(num_qubits):
        qc.ry(x[i],[i])
    for i in range(num_qubits):
        qc.rz(x[num_qubits+i],[i])
    former = int((num_qubits) /2)
    later = int((num_qubits-1) /2)
    for i in range(former):
        qc.cz([2*i+1],[2*i])
    for i in range(num_qubits):
        qc.ry(x[2*num_qubits + i],[i])
    for i in range(num_qubits):
        qc.rz(x[3*num_qubits+i],[i])
    for i in range(later):
        qc.cz([2*i+2],[2*i+1])
    encode = qc
    return [encode,num_qubits]

def kernel_J(x,repeat=1):
    r = 4
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel J")
    for i in range(num_qubits):
        qc.ry(x[i],[i])
    for i in range(num_qubits-1):
        qc.crz(x[num_qubits+i],[i],[i+1])
    qc.crz(x[2*num_qubits-1],num_qubits-1,0)
    for i in range(num_qubits):
        qc.ry(x[2*num_qubits + i],[i])
    for i in range(num_qubits-1):
        qc.crz(x[3*num_qubits+i],[i+1],[i])
    qc.crz(x[4*num_qubits-1],0,num_qubits-1)
    encode = qc
    return [encode,num_qubits]

def kernel_K(x,repeat=1):
    r = 4
    n = len(x)
    num_qubits = int(((n-1)+ (r-(n-1)%r))/r)
    x = x+np.zeros(r*num_qubits-n).tolist()
    qc = QuantumCircuit(num_qubits,name = "Kernel K")
    for i in range(num_qubits):
        qc.ry(x[i],[i])
    
    for i in range(num_qubits-1):
        qc.crx(x[num_qubits+i],[i],[i+1])
    qc.crx(x[2*num_qubits-1],num_qubits-1,0)
    for i in range(num_qubits):
        qc.ry(x[2*num_qubits + i],[i])
    for i in range(num_qubits-1):
        qc.crx(x[3*num_qubits+i],[i+1],[i])
    qc.crx(x[4*num_qubits-1],0,num_qubits-1)
    encode = qc
    return [encode,num_qubits]

