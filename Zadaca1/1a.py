import numpy as np
import matplotlib.pyplot as plt
import signal
import csv

def funkcijaA(x):
    fx = 0.2 * (x - 2.0) * np.sin(7.0 * x - 15.0) + 1
    mask = fx >= 1.0
    fx_out = np.zeros_like(x)  
    fx_out[mask] = np.cos(2.0 / (np.power(x[mask] - 6.0, 2.0) + 1.0)) + 0.7
    fx_out[~mask] = fx[~mask]
    return fx_out

class PSO:
    def __init__(self, size, lower_bound, upper_bound, inertia, cognitive, social):
        self.Xlb = lower_bound
        self.Xub = upper_bound
        self.swarm_size = size
        self.w = inertia
        self.C1 = cognitive
        self.C2 = social

        self.x = np.random.uniform(self.Xlb, self.Xub, self.swarm_size)
        self.v = np.zeros(self.swarm_size)
        self.x_PB = self.x.copy()
        self.x_NB = None
        self.fx_NB = float('inf')
        self.iterations_data = [] 

    def run(self, n_iterations):
        for iter in range(n_iterations):
            iteration_data = []  
            for i in range(self.swarm_size):
                if self.x_NB is None:
                    self.x_NB = self.x[i]
                self.v[i] = self.w * self.v[i] + self.C1 * np.random.random() * (self.x_PB[i] - self.x[i]) + \
                        self.C2 * np.random.random() * (self.x_NB - self.x[i])
                self.x[i] += self.v[i]

                if self.x[i] > self.Xub:
                    self.x[i] = 2 * self.Xub - self.x[i]
                    self.v[i] *= -1  
                elif self.x[i] < self.Xlb:
                    self.x[i] = 2 * self.Xlb - self.x[i]
                    self.v[i] *= -1  

                fx = funkcijaA(self.x[i])

                if fx < funkcijaA(self.x_PB[i]):
                    self.x_PB[i] = self.x[i]

                if fx < self.fx_NB:
                    self.x_NB = self.x[i]
                    self.fx_NB = fx

                iteration_data.append((iter, self.x[i], fx))  
            self.iterations_data.append(iteration_data)  

    def print_result(self):
        print("Najbolje pronadeno rjesenje: x_NB =", self.x_NB, ", f(x_NB) =", self.fx_NB)

    def save_to_csv(self, filename='rezultati.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'X', 'f(X)'])  
            for iteration_data in self.iterations_data:
                writer.writerows(iteration_data)

def signal_handler(signal, frame):
    print("Izvođenje programa prekinuto.")
    plt.close()  
    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    pso = PSO(10, 2.0, 6.0, 0.6, 1.2, 1.8)
    pso.run(100)
    pso.print_result()
    pso.save_to_csv()

    x_values = np.linspace(2.0, 6.0, 1000)  
    y_values = funkcijaA(x_values)
    xNB = pso.x_NB
    fxNB = pso.fx_NB

    plt.plot(x_values, y_values, label='funkcijaA(x)')
    plt.scatter(xNB, fxNB, color='red', label='Najbolje rješenje') 
    plt.xlabel('x - vrijednosti')
    plt.ylabel('funkcijaA(x)')
    plt.title('Graf funkcije funkcijaA s najboljim rješenjem')
    plt.legend()
    plt.grid(True)
    plt.show()
