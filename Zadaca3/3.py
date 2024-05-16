import numpy as np
import pandas as pd
import concurrent.futures

def load_qap(filename):
    with open(filename, 'r') as file:
        n = int(file.readline())
        a = np.zeros((n, n)) 
        for i in range(n):
            values = list(map(int, file.readline().split()))
            a[i, :len(values)] = values  
        
    return a

def initialize_pheromone_matrix(n):
    return np.ones((n, n))

def generate_initial_solution(n):
    return np.random.permutation(n) if np.random.rand() > 0.1 else np.arange(n)

def calculate_cost(solution, flow_matrix, distance_matrix):
    # Provjera jesu li matrice istih dimenzija
    if flow_matrix.shape != distance_matrix.shape:
        raise ValueError("Matrice flow_matrix i distance_matrix moraju biti istih dimenzija.")
    
    # Provjera jesu li matrice kvadratne
    if flow_matrix.shape[0] != flow_matrix.shape[1]:
        raise ValueError("Matrice flow_matrix i distance_matrix moraju biti kvadratne.")

    # Provjeravanje ispravnosti indeksa u rješenju
    if np.min(solution) < 0 or np.max(solution) >= len(solution):
        raise ValueError("Indeksi u rješenju nisu u ispravnom rasponu.")

    return np.sum(flow_matrix * distance_matrix[solution][:, solution])

def local_search(solution, flow_matrix, distance_matrix):
    improved = True
    while improved:
        improved = False
        for i in range(len(solution) - 1):
            for j in range(i + 1, len(solution)):
                new_solution = solution.copy()
                new_solution[i:j+1] = np.flip(solution[i:j+1])
                new_cost = calculate_cost(new_solution, flow_matrix, distance_matrix)
                if new_cost < calculate_cost(solution, flow_matrix, distance_matrix):
                    solution = new_solution
                    improved = True
    return solution

def update_pheromones(pheromone_matrix, solutions, costs, tau_max, tau_min, evaporation_rate=0.1):
    num_ants = len(solutions)
    for ant in range(num_ants):
        for i in range(len(solutions[ant]) - 1):
            for j in range(i + 1, len(solutions[ant])):
                pheromone_deposit = 1 / costs[ant]
                # Ažuriranje gornjeg trokuta matrice
                pheromone_matrix[i, j] = (1 - evaporation_rate) * pheromone_matrix[i, j] + pheromone_deposit
                # Ovdje ažuriramo odrazni element u donjem trokutu matrice
                pheromone_matrix[j, i] = pheromone_matrix[i, j]
    return pheromone_matrix

def parallel_local_search(args):
    solution, flow_matrix, distance_matrix = args
    return local_search(solution, flow_matrix, distance_matrix)

def MMAS_QAP(flow_matrix, distance_matrix, num_ants, max_iterations, rho):
    n = len(flow_matrix)
    tau_max = 1 / (rho * calculate_cost(np.arange(n), flow_matrix, distance_matrix))
    tau_min = tau_max * pow((1.0 - pow(0.05, 1.0/n)) / ((n/2.0 - 1.0)*pow(0.05, 1.0/n)), 1.0)
    pheromone_matrix = np.full((n, n), tau_max)
    best_solution = None
    best_cost = float('inf')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for iteration in range(max_iterations):
            solutions = np.array(list(executor.map(generate_initial_solution, [n] * num_ants)))
            solutions = np.array(list(executor.map(local_search, solutions, [flow_matrix] * num_ants, [distance_matrix] * num_ants)))
            costs = np.array(list(executor.map(calculate_cost, solutions, [flow_matrix] * num_ants, [distance_matrix] * num_ants)))
            
            min_cost_index = np.argmin(costs)
            if costs[min_cost_index] < best_cost:
                best_solution = solutions[min_cost_index]
                best_cost = costs[min_cost_index]
            
            pheromone_matrix = update_pheromones(pheromone_matrix, solutions, costs, tau_max, tau_min)
            
            # Dodajmo ispisnu poruku koja prikazuje trošak za svako rješenje
            print(f"Iteracija: {iteration + 1}, Troškovi rješenja: {costs}")
            
            # Dodajmo i ispis najboljeg rješenja i najboljeg troška
            print(f"Najbolje rješenje: {best_solution}, Najbolji trošak: {best_cost}")
    
    return best_solution, best_cost

def test_algorithm(problem_choice, algorithm):
    if problem_choice == 0:
        return  # Izađi iz funkcije ako je odabran izlaz
    elif problem_choice == 1:
        problem_name = 'had12'
        flow_matrix = load_qap('had12.dat')
    elif problem_choice == 2:
        problem_name = 'els19'
        flow_matrix = load_qap('els19.dat')
    elif problem_choice == 3:
        problem_name = 'tho40'
        flow_matrix = load_qap('tho40.dat')
    else:
        print("Nepoznat problem!")
        return
    
    n = len(flow_matrix)  # Broj redova u matrici
    max_iterations = 10
    rho = 0.01
    num_ants = 100
    best_solution, best_cost = algorithm(flow_matrix, flow_matrix, num_ants, max_iterations, rho)
    print(f"Najbolje rješenje za {problem_name}: {best_solution}, cost: {best_cost}")

    results_df = pd.DataFrame({'Iteration': range(1, max_iterations + 1), 'Best Cost': best_cost})
    results_df.to_csv(f'{problem_name}_results.csv', index=False)

    # Rekurzivno pozivanje za novi odabir problema
    problem_choice = int(input("Odaberite problem (1 - HAD12, 2 - ELS19, 3 - THO40, 0 - Izađi): "))
    test_algorithm(problem_choice, algorithm)

problem_choice = int(input("Odaberite problem (1 - HAD12, 2 - ELS19, 3 - THO40, 0 - Izađi): "))
test_algorithm(problem_choice, MMAS_QAP)


