# mdp

# Table of contents
1. [Solution for the finite-time](#Solution-for-the-finite-time)
2. [Solution for the infinite-time](#Solution-for-the-infinite-time)


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

The solution for the infinite-time discrete optimal control problem with functions that describe dynamics of the analytical data system.
Files for this task are stores in lab2.
```
usage: python optimal.py [-h] [--path PATH] [--eps EPS]

options:
  -h, --help   show this help message and exit
  --path PATH  path to your csv file
  --eps EPS    epsilon to reset small values
```