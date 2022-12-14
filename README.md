# mdp

# Table of contents
1. [Solution for the finite-time](#Solution-for-the-finite-time)
2. [Solution for the infinite-time](#Solution-for-the-infinite-time)
3. [Solution for the infinite-time stochastic problem](#Solution-for-the-infinite-time-stochastic-problem)
4. [Reduction of the MDP to a stochastic discrete optimal control](#Reduction-of-the-MDP-to-a-stochastic-discrete-optimal-control)
5. [Solution for the stochastic discrete optimal control with discount](#Solution-for-the-stochastic-discrete-optimal-control-with-discount)
6. [Solution for the discounted Markov decision problem](#Solution-for-the-discounted-Markov-decision-problem)

## Solution for the finite-time

The solution for the finite-time discrete optimal control problem with functions that describe dynamics of the analytical data system.
Files for this task are stores in lab1.
```
usage: python optimal.py [-h] [--tmin TMIN] [--tmax TMAX] [--x0 X0] [--xf XF] [--print] [--path PATH]

options:
  -h, --help   show this help message and exit
  --tmin TMIN  the minimal time from which the cost is calculated
  --tmax TMAX  the maximum time until which the cost is calculated
  --x0 X0      node to start
  --xf XF      node to finish
  --print      print pretty table for you
  --path PATH  path to your csv file
```

## Solution for the infinite-time

The solution for the infinite-time discrete optimal control problem
with functions that describe dynamics of the analytical data system.
Files for this task are stores in lab2.
```
usage: python optimal.py [-h] [--path PATH] [--print]

options:
  -h, --help   show this help message and exit
  --path PATH  path to your csv file
  --print      print cost table
```

## Solution for the infinite-time stochastic problem
```
usage: python optimal.py [-h] [--path PATH] [--print]

options:
  -h, --help   show this help message and exit
  --path PATH  path to your csv file
  --print      print cost table
```

## Reduction of the MDP to a stochastic discrete optimal control
```
usage: python optimal.py [-h] [--path PATH]

options:
  -h, --help   show this help message and exit
  --path PATH  path to your csv file
```

## Solution for the stochastic discrete optimal control with discount
In this section we will examine the stochastic discrete 
optimal control
problem over infinite time intervals and with the discounted 
summary cost optimization criterion.
```
usage: python optimal.py [-h] [--x0 X0] [--gamma GAMMA] [--path PATH] [--print]

options:
  -h, --help     show this help message and exit
  --x0 X0        node to start
  --gamma GAMMA  gamma discount
  --path PATH    path to your csv file
  --print        print cost table
```

## Solution for the discounted Markov decision problem
```
usage: python optimal.py [-h] [--gamma GAMMA] [--path PATH]

options:
  -h, --help     show this help message and exit
  --gamma GAMMA  gamma discount
  --path PATH    path to your csv file
```