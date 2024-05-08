from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np

#Constants
m = 2.88*(10**(-10)) #kg. effective mass of moving elements
w_r = 44700 #rad/s. natural frequency of the resonant mode
E_r = 0.89 # damping ratio of resonant mode
w_t = 195000 #rad/s. natural frequency of resonant mode
E_t = 1.23 # dampling ratio of resonant mode
k = 0.576 #N/m. End spring constant k1 & k2
c = 1.15*(10**(-5)) #Ns/m. Dash-pot damping constant
c_3 = 2.88*(10**(-5)) #Ns/m. Dash-pot damping constant
k_3 = 5.18 #N/m. Coupling spring constant
s = 0.288*(10**(-6)) #m^2. Surface area of membranes
# J = time for incident to travel between the points where the forces act.
#   dsin(incident_angle)/c where:
d = 1.2 #mm. Distance between force points
#c = 344 #m/s. speed of sound in real life

#Initial conditions
x1_0 = 0
x2_0 = 0
x1_dot_0 = 0
x2_dot_0 = 0

M = np.array([[m, 0], [0, m]])
C = np.array([[(c + c_3), c_3], [c_3, (c + c_3)]])
K = np.array([[(k + k_3), k_3], [k_3, (k + k_3)]])

def external_force(t):
    frequency = 5000
    amplitude = 1.0*(10**(-8))
    if t <= 0.5:
        return amplitude * np.sin(2 * np.pi * frequency * t)
    else:
        return 0

def external_force_shifted(t):
    frequency = 5000
    amplitude = 1.0*(10**(-8))
    time_delay = 1.2*(10**(-3))*np.sin(45*np.pi/180)/344
    if t <= 0.5 + time_delay:
        return amplitude * np.sin(2 * np.pi * frequency * (t - time_delay))
    else:
        return 0

t = np.linspace(0, 1, 1600000) #1,600,000 time steps from 0 to 1

#Simulated sine waves arriving at different times.
#Use actual microphone data as input here
f1 = np.array([external_force(ti) for ti in t])
f2 = np.array([external_force_shifted(ti) for ti in t])
f = np.vstack((f1,f2)).T

M_inv = np.linalg.inv(M)

#x_dot_dot = np.dot(M_inv, (f - np.dot(C, x_dot) - np.dot(K, x)))
#Initialise arrays to store displacement and velocity
x = np.zeros((len(t), 2))
x_dot = np.zeros((len(t), 2))

#Initial conditions
x[0] = [x1_0, x2_0]
x_dot[0] = [x1_dot_0, x2_dot_0]

#Time step size
dt = (t[1] - t[0])

#Perform Euler integration
for i in range(1, len(t)):
    x_dot_dot = np.dot(M_inv, (f[i] - np.dot(C, x_dot[i-1]) - np.dot(K, x[i-1])))
    x_dot[i] = x_dot[i-1] + x_dot_dot * dt
    x[i] = x[i-1] + x_dot[i] * dt

x1 = x[:,0]
x2 = x[:,1]

plt.plot(t, x1, label='Displacement x1')
plt.plot(t, x2, label='Displacement x2')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.grid(True)
plt.show()

