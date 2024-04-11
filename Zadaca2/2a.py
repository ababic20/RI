import numpy as np
from scipy.optimize import differential_evolution
import csv
import matplotlib.pyplot as plt

def error_function(coef, *args):
    x_values, y_values = args[0], args[1]
    y_pred = np.polyval(coef, x_values)
    error = np.sum((y_pred - y_values) ** 2)
    return error

def input_points():
    points = []
    for i in range(5):
        coord = input(f"Unesite koordinate {i+1}. točke u formatu x y: ")
        x, y = map(float, coord.split())
        points.append((x, y))
    return points

print("Unesite koordinate 5 točaka:")
print("Npr. točke iz pdfa su: 1 1, 2.2 3.9, -0.8 -11, 1.5 1.5, 5.2 103")
print("Unosi se u formatu npr: 1 1")
points = input_points()
x_values = [point[0] for point in points]
y_values = [point[1] for point in points]

def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteracija', 'a3', 'a2', 'a1', 'a0', 'Pogreška'])
        for row in data:
            writer.writerow(row)

# Funkcija za praćenje napretka algoritma
def callback_func(xk, convergence):
    global iterations, results, coef_changes, error_changes
    iterations += 1
    error = error_function(xk, x_values, y_values)
    results.append([iterations, *xk, error])
    print(f"Iteracija: {iterations}, Pogreška: {error}")

    # Dodavanje podataka za praćenje promjene koeficijenata i pogreške
    coef_changes.append(xk)
    error_changes.append(error)

iterations = 0
results = []
coef_changes = []
error_changes = []

# Pokretanje diferencijalne evolucije
result = differential_evolution(error_function, bounds=[(-10, 10)]*4, args=(x_values, y_values), strategy='best1bin', callback=callback_func)

print("\nOptimalni koeficijenti:")
print(result.x)
write_to_csv('de_results.csv', results)

# Funkcija za izračunavanje vrijednosti krivulje
def curve(x, coef):
    return np.polyval(coef, x)


x_values_plot = np.linspace(min(x_values), max(x_values), 100)
y_values_plot = curve(x_values_plot, result.x)

plt.figure(figsize=(10, 8))

# Grafikon za prikaz točaka i krivulje
plt.subplot(3, 1, 1)
plt.scatter(x_values, y_values, color='red', label='Zadane točke')
plt.plot(x_values_plot, y_values_plot, color='blue', label='Krivulja')
plt.title('Krivulja koristeći diferencijalnu evaluaciju')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Grafikon za prikaz promjene pogreške kroz iteracije
plt.subplot(3, 1, 2)
iterations_range = range(1, iterations + 1)
plt.plot(iterations_range, error_changes, color='red', linestyle='--', label='Pogreška')
plt.title('Promjena pogreške kroz iteracije')
plt.xlabel('Iteracija')
plt.ylabel('Pogreška')
plt.grid(True)
plt.legend()

# Grafikon za prikaz promjene koeficijenata kroz iteracije
plt.subplot(3, 1, 3)
plt.plot(iterations_range, [coef[0] for coef in coef_changes], label='a3')
plt.plot(iterations_range, [coef[1] for coef in coef_changes], label='a2')
plt.plot(iterations_range, [coef[2] for coef in coef_changes], label='a1')
plt.plot(iterations_range, [coef[3] for coef in coef_changes], label='a0')
plt.title('Promjena koeficijenata kroz iteracije')
plt.xlabel('Iteracija')
plt.ylabel('Vrijednost koeficijenta')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
