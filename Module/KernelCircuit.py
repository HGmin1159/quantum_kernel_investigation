import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

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
