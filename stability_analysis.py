import numpy as np
import matplotlib.pyplot as plt
import control.matlab as cm
import control

## Example 3.1 converting ODE to State-space form
## from Feedback System by Alstrom and Murray

## My Spring-Mass system equation:
#   m * (d^2/dt^2)q + c * (d/dt)q + k * q = u (eqs. 3.7)
print("My Spring-Mass system equation: \n",
      "m * (d^2/dt^2)q + c * (d/dt)q + k * q = u \n")

## States of the system:
#   x_1 = (d/dt)q, x_2 = q

## Derivative of states:
#   (d/dt)x_1 = (d^2/dt^2)q
#   (d/dt)x_1 = (-c / m) * (d/dt)x_1 + (-k / m) * x_2 + (1 / m) * u
#   (d/dt)x_2 = (d/dt)q = x_1

## State-space representation (d/dt)x = A * x + B * u:
#   [dx / dt] = [[(dx1 / dt)],
#                [(dx2 / dt)]]
#   [u] = u
#   A = [[(-c / m), (-k / m)],
#        [ 1, 0]]
#   B = [[(1 / m)],
#        [0]]

## For example, the value [m, c, k] = [2, 0.5, 3].
m, c, k = (2, 0.5, 3)
print("For example, \n m = %d, c = %d, k = %d. \n" % (m, c, k))

## Write down the A matrix and B matrix.
A = np.array(
    [[(-c / m), (-k / m)],
     [1, 0]]
    )
B = np.array(
    [[(1 / m), 0],
     [0, 1]]
    )

## Take the outputs = states, so C is identity matrix. Set D to 0.
C = np.identity(np.ndim(A))
D = np.zeros((np.shape(C)[0], np.shape(B)[1]))

my_sys = cm.ss(A, B, C, D)
print("""Assume the outputs are all states.
My Spring-Mass system continuous time state-space representation: \n""", my_sys)

## The stability of my Spring-Mass system can be represented by the poles location
poles = cm.pole(my_sys)
print("The poles locations are: \n", poles)
pole, zero = control.pzmap(my_sys)
plt.show()
print("The poles real parts are negative. Thus, my Spring-Mass system is stable.")
