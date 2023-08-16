```
mpc_code
├─ expect.exp
├─ Input
│  ├─ Input-P0-0
│  ├─ Input-P1-0
│  ├─ input_mnist.py
│  └─ test_cifar10.py
├─ ip.txt
├─ mpc_deploy.sh
├─ README.md
├─ run.exp
└─ Source
   ├─ 2pcaes_cifar10.mpc
   ├─ 2pcaes_mnist.mpc
   ├─ end_cifar10.mpc
   └─ end_mnist.mpc

```

### Overview

The implement of AES encryption and Shapley value calculation using secure multi-party computation (MPC) protocols with SPDZ2k, which is base on the `MP-SPDZ` library. 

### Requirements

- Refer to the MP-SPDZ  https://github.com/data61/MP-SPDZ
- Place the `Source` folders in the same name folder in MP-SPDZ.
- Place the use the input code to generate the input data.

### Data preparation
The input data comes from the model parameters and the transformed results of the dataset, which have been submitted in other files. A conversion code demo is provided here, which can generate the input file. A simple input data is also provided here. 
**2pc**

- `Input-P0-0`:Model parameters
- `Input-P1-0`:the image data

**mpc**

- `Input-P0-0`:Model parameters and the secret shared result of `2pc` with P1,P2,P3...
- `Input-P1-0`:The secret shared result `2pc` with P0.
- `...`

### Complication

Command line: 

`$ ./compile.py -R 64 <program>`
e.g.
`$ ./compile.py -R 64 2pcaes_cifar10`

`$ ./compile.py -R 64 end_cifar10`

### Running
**2pc**

- Server 0: model owner 
  `$ ./apdz2k-party.x -N 2 -h ip_0 0 2pcaes_mnist`

- Server 1: data owner 
  `$ ./apdz2k-party.x -N 2 -h ip_0 1 2pcaes_mnist`

**mpc**

- Server 0: model owner 
- Other Server: data owner 
  `$ ./apdz2k-party.x -N <the count of servers> -h ip_0 0 end_cifar10`
  `$ ./apdz2k-party.x -N <the count of servers> -h ip_0 1 end_cifar10`
  `$ ./apdz2k-party.x -N <the count of servers> -h ip_0 2 end_cifar10`
  `...`

You can use the `mpc_deploy.sh` to run the mpc phase code. You need to write the servers ip in the ip.txt and then run the shell code.

