import numpy as np
from qiskit import  QuantumCircuit,transpile,Parameter
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.providers.aer import QasmSimulator


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


def kernel_circuit_drawer(kernel,data=8):
    if type(data) == int : 
        theta = []
        for i in range(data):
            theta = theta+[Parameter("theta"+str(i))]
    else : theta = data
    gate,num_qubit = kernel(theta)
    qc = QuantumCircuit(num_qubit)
    qc.append(gate,range(num_qubit))
    p1 = qc.draw("mpl")
    return(p1)
