import numpy as np
import pandas as pd

def load_qap(filename):
    with open(filename, 'r') as file:
        n = int(file.readline())
        a = np.zeros((40, 40)) 
        for i in range(n):
            values = list(map(int, file.readline().split()))
            a[i, :len(values)] = values  
        
    return a

def initialize_pheromone_matrix(n):
    return np.ones((n, n))

def generate_initial_solution(n):
    return np.random.permutation(n)

def calculate_cost(solution, flow_matrix, distance_matrix):
    cost = 0
    for i in range(len(solution)):
        for j in range(len(solution)):
            cost += flow_matrix[i, j] * distance_matrix[solution[i], solution[j]]
    return cost

# MMAS algoritam za rješavanje QAP problema
def MMAS_QAP(flow_matrix, distance_matrix, num_ants, max_iterations, rho):
    n = len(flow_matrix)
    tau_max = 1 / (rho * calculate_cost(np.arange(n), flow_matrix, distance_matrix))
    tau_min = tau_max * pow((1.0 - pow(0.05, 1.0/n)) / ((n/2.0 - 1.0)*pow(0.05, 1.0/n)), 1.0)
    pheromone_matrix = np.full((n, n), tau_max)
    best_solution = None
    best_cost = float('inf')
    for iteration in range(max_iterations):
        solutions = np.zeros((num_ants, n), dtype=int)
        costs = np.zeros(num_ants)
        
        for ant in range(num_ants):
            solution = generate_initial_solution(n)
            local_search(solution, flow_matrix, distance_matrix)
            solutions[ant] = solution
            costs[ant] = calculate_cost(solution, flow_matrix, distance_matrix)
        
        min_cost_index = np.argmin(costs)
        if costs[min_cost_index] < best_cost:
            best_solution = solutions[min_cost_index]
            best_cost = costs[min_cost_index]
        
        
        pheromone_matrix = update_pheromones(pheromone_matrix, solutions, costs, tau_max, tau_min)
        
    return best_solution, best_cost

def local_search(solution, flow_matrix, distance_matrix):
    pass

def update_pheromones(pheromone_matrix, solutions, costs, tau_max, tau_min):
    pass

def test_algorithm(problem_name, algorithm):
    if problem_name == 'had12':
        flow_matrix = load_qap('had12.dat')
    elif problem_name == 'els19':
        flow_matrix = load_qap('els19.dat')
    elif problem_name == 'tho40':
        flow_matrix = load_qap('tho40.dat')
    else:
        print("Nepoznat problem!")
        return
    num_ants = len(flow_matrix)
    max_iterations = 100 
    rho = 0.02
    best_solution, best_cost = algorithm(flow_matrix, flow_matrix, num_ants, max_iterations, rho)
    print(f"Najbolje rješenje za {problem_name}: {best_solution}, cost: {best_cost}")

    results_df = pd.DataFrame({'Iteration': range(1, max_iterations + 1), 'Best Cost': best_cost})
    results_df.to_csv(f'{problem_name}_results.csv', index=False)

test_algorithm('had12', MMAS_QAP)
test_algorithm('els19', MMAS_QAP)
test_algorithm('tho40', MMAS_QAP)
