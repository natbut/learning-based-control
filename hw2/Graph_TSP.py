
import csv
import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import numpy as np
import math
from itertools import permutations
import time

class Graph_TSP_Solver():

    def __init__(self):
        # self.G = nx.DiGraph()
        # self.G = nx.Graph()
        self.cities = []
        self.weights = {}

    # Generate random nodes
    def generate_random_tsp(self, num_cities):
        cities = self.cities
        for i in range(num_cities):
            x, y = random.randint(50, 450), random.randint(50, 450)
            cities.append((x, y))
    
    # Generate nodes from file
    def generate_from_file(self, filepath):
        cities = self.cities

        try:
            with open(filepath, mode='r', newline='') as file:
                csv_reader = csv.reader(file)
                
                for row in csv_reader:
                    if len(row) == 2:  # Ensure each row has exactly 2 columns
                        try:
                            x = float(row[0])
                            y = float(row[1])
                            cities.append((x, y))
                        except ValueError:
                            print(f"Skipping invalid row: {row}")
                    else:
                        print(f"Skipping row with incorrect number of columns: {row}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"An error occurred: {e}")
 
    # Create nodes & generate edges
    def display_tsp_graph(self, tour=None):
        G = nx.Graph()
        num_cities = []
        cities = self.cities

        for i, (x, y) in enumerate(cities):
            G.add_node(i, pos=(x, y))
            num_cities.append(i)

        # generate edges
        for i in range(len(cities)):
            for j in num_cities:
                if j == i: continue
                city1 = cities[i]
                city2 = cities[j]
                distance = round(((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)**0.5, 4)
                G.add_edge(i, j, weight=distance)
                self.weights[(i,j)] = distance # create weights dictionary

        return G

    # Calculate the total distance for a TSP tour
    def calculate_total_distance(self, tour):
        weights = self.weights
        total_distance = 0
        for i in range(len(tour) - 1):
            city1_idx = tour[i]
            city2_idx = tour[i + 1]
            distance = weights[(city1_idx, city2_idx)]
            total_distance += distance
        return total_distance


    # Swap 2 elements random in list
    def swap_two_random_elements(self, init_list):
        idx_a = random.randint(0,len(init_list)-1)
        idx_b = len(init_list) - 1 - idx_a
        a = init_list[idx_a]
        b = init_list[idx_b]
        swapped_list = init_list.copy()
        swapped_list[idx_a] = b
        swapped_list[idx_b] = a
        
        return swapped_list
    
    # Swap 2 elements at given indices
    def swap_two_elements(self, idx_a, idx_b, init_list):
        swapped_list = init_list.copy()
        a = init_list[idx_a]
        b = init_list[idx_b]
        swapped_list[idx_a] = b
        swapped_list[idx_b] = a
        return swapped_list

    # Solve TSP using simulated annealing approach
    def tsp_sa(self, t=10, alpha = 0.9, steps = 100):
        best_tour = None
        solutions_count = 0
        performance_vals = []

        # 1) Start from initial (random) state
        current = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,
                            14,15,16,17,18,19,20,21,22,23,24])
        np.random.shuffle(current)

        # best
        current_tour = [0] + current.tolist() + [0]
        current_performance = self.calculate_total_distance(current_tour)
        solutions_count = 1
        performance_vals.append(current_performance)

        STEPS = steps
        T = t 
        ALPHA = alpha
        step = 0
        while step < STEPS:
            # 2) Update parameter T (temperature)
            #   - T = T/(iteration_number+1)
            T = T*ALPHA  #round(T/(step+1),6)  # T/2 # T/(step+1)

            # 3) Randomly generate successor state from current state
            #   - swap 2 elements at random

            successor = self.swap_two_random_elements(current)

            successor_tour = [0] + successor.tolist() + [0]
            successor_performance = self.calculate_total_distance(successor_tour)

            # 4) If successor is better than current:
            #   - select it as next state
            #   - return to step 2
            if successor_performance < current_performance:
                current = successor
                current_performance = successor_performance
                
            # 5) If successor is NOT better than current:
            #   - Select it with probability p = exp(-dE/T)
            #       - dE = objective(new) - objective(current)
            #   - return to step 2 
            # Objective: minimize total distance
            else:
                dE = successor_performance - current_performance 
                test = random.random()
                # avoid divide by 0 error when T -> 0
                if T == 0.000000: p =  0
                else: p = math.exp(-dE/T)
                #p = math.exp(-dE/T)
                if test <= p:
                    current = successor
                    current_performance = successor_performance
            step += 1
            solutions_count += 1

            performance_vals.append(current_performance)

        best_tour = [0] + current.tolist() + [0]

        return best_tour, solutions_count, performance_vals


    def tsp_evo(self, k, n, alpha, steps):
        best_tour = None
        solutions_count = 0
        performance_vals = []

        STEPS = steps

        # 1) Start from k initial (random) states
        states = []
        for i in range(k):
            st = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,
                            14,15,16,17,18,19,20,21,22,23,24])
            np.random.shuffle(st)
            states.append(st.tolist())

        # Tracking stuff
        solutions_count = k
        step = 0
        
        while step < STEPS:
            solutions_count += k
            # 2) Generate k successor states randomly
            # 3) Perturb those states (mutation)
            swapped_states = []
            for state in states:
                # do n swaps
                swapped = self.swap_two_random_elements(state)
                i = 0
                while i < n-1:
                    swapped = self.swap_two_random_elements(swapped)
                    i += 1

                swapped_states.append(swapped)

            states = states + swapped_states.copy()

            tours = []
            tour_dists = []
            for state in states:
                tour = [0] + state + [0]
                tours.append(tour)
                dist = round(self.calculate_total_distance(tour), 5)
                tour_dists.append(dist)

            # 4) Select k states from the pool of 2k states, with value of state affecting 
            # its selection probability
                # -  roulette wheel method

            # Inverse probabilities calculated, then normalized in order to give smaller values higher probs
            dist_sum = sum(tour_dists)
            norm_vals = []
            for dist in tour_dists:
                norm_vals.append(round((dist/dist_sum), 5))

            inverse = []
            for val in norm_vals:
                inverse.append((1.0/val)**alpha)

            inv_sum = sum(inverse)
            probabilities = []
            for val in inverse:
                probabilities.append(round(val/inv_sum, 5))

            cum_probs = [probabilities[0]]
            for i in range(1,len(probabilities)):
                cum_probs.append(cum_probs[i-1] + probabilities[i])

            # Selection process
            new_states = []
            while len(new_states) < k: # keep k states
                choose = 0
                random_num = round(random.random(), 6)
                for i in range(len(probabilities)):
                    choose += probabilities[i]
                    if choose >= random_num:
                        if states[i] not in new_states: #only add new states
                            new_states.append(states[i])
                            #print('WITH ', random_num, ' SELECTED ', tour_dists[i])
                        break

            # 5) Go to step 2
            states = new_states.copy()
            step += 1

            # Process best tour for performance tracking
            best_tour = tours[0]
            best_val = self.calculate_total_distance(best_tour)
            for tour in tours[1:]:
                val = self.calculate_total_distance(tour)
                if val < best_val:
                    best_tour = tour
                    best_val = val
            performance_vals.append(best_val)

        # Process final output tour
        best_tour = tours[0]
        best_val = self.calculate_total_distance(best_tour)
        for tour in tours[1:]:
            val = self.calculate_total_distance(tour)
            if val < best_val:
                best_tour = tour
                best_val = val
        performance_vals.append(best_val)

        return best_tour, solutions_count, performance_vals

    # beam search
    def tsp_pop(self, k=2, steps=100):
        best_tour = None
        solutions_count = 0
        performance_vals = []

        STEPS = steps
        K = k

        # 1) Start from k initial (random) states
        states = []
        for i in range(K):
            st = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,
                            14,15,16,17,18,19,20,21,22,23,24])
            np.random.shuffle(st)
            states.append(st.tolist())

        solutions_count = k
        step = 0
        
        while step < STEPS:

            # 2) Generate all successor states for all current states
            new_states = [] # states to add to existing list of states
            for state in states:
                # add to states all new states generated by swapping
                    # 2 elements in state
                generated_states = [] # new states generated from each state
                for i in range(len(state)-1):
                    # generate new lists by swapping elem with all remaining elements
                    for j in range(i+1, len(state)): #for each element in rest of list
                        # create new list with element at i swapped with element at j
                        new_state = self.swap_two_elements(i, j, state)
                        # add list to new
                        generated_states.append(new_state)
                        solutions_count += 1

                new_states += generated_states.copy()
            states += new_states.copy()

            # 3) Select k best states from pool
            # Evaluate value of each state
            tours = []
            tour_dists = []
            for state in states:
                tour = [0] + state + [0]
                tours.append(tour)
                dist = round(self.calculate_total_distance(tour), 10)
                tour_dists.append(dist)

            # Gest best tours & indices of these tours
            best_states_values = self.get_k_min_elements_indices(tour_dists, K)

            # Select states to keep
            keep_states = []
            for res in best_states_values:
                keep_states.append(states[res])

            # 4) Back to step 2
            states = keep_states
            step += 1
            # Process best tour for performance tracking
            best_tour = tours[0]
            best_val = self.calculate_total_distance(best_tour)
            for tour in tours[1:]:
                val = self.calculate_total_distance(tour)
                if val < best_val:
                    best_tour = tour
                    best_val = val
            performance_vals.append(best_val)
            
        # Process final output tour
        best_tour = tours[0]
        best_val = self.calculate_total_distance(best_tour)
        for tour in tours[1:]:
            val = self.calculate_total_distance(tour)
            if val < best_val:
                best_tour = tour
                best_val = val

        return best_tour, solutions_count, performance_vals
    
    def get_k_min_elements_indices(self, lst, k):
        indices = sorted(range(len(lst)), key=lambda x: lst[x])[:k] # return list containing indices of k largest elements
        return indices

    
    def test_tsp_sa(self, tests, T, alpha, steps):
        print('T: ', T, ' Alpha: ', alpha, ' Step: ', steps)
        print('==============================')

        filename = 'test_sa_T' + str(T) + '_alpha' + str(alpha) + '_steps' + str(steps) + '.txt'

        start = time.time()
        dists = []
        with open(filename, 'w') as file:
            file.write('T' + str(T) + '_alpha' + str(alpha) + '_steps' + str(steps))
            for i in range(tests):
                tour, solutions_count, performance = self.tsp_sa(T, alpha, steps)
                dist = round(self.calculate_total_distance(tour), 5)
                dists.append(dist)
                print('TEST ', i, ' |  Dist: ', dist, 
                    ' | Solutions Explored: ', solutions_count, 
                    '\n Tour: ', tour)

                content = '\nTEST ' + str(i) + ' |  Dist: ' + str(dist) +  ' | Solutions Explored: ' + str(solutions_count) + ' Tour: ' + str(tour) + '\nPerf Track: ' + str(performance)

                file.write(content)

            end = time.time()
            file.write('\nAvg Runtime: '+ str((end-start)/tests))
            file.write('\nAvg Dist: ' + str(sum(dists)/tests))
    
    
    def test_tsp_evo(self, tests, k, n, alpha, steps):
        print('k: ', k, ' n: ', n, ' alpha: ', alpha, ' step: ', steps)
        print('==============================')

        filename = 'test_evo_k' + str(k) + '_n' + str(n) + '_alpha' + str(alpha) + '_steps' + str(steps) + '.txt'

        start = time.time()
        dists = []
        with open(filename, 'w') as file:
            file.write('k: ' + str(k) + ' n: ' + str(n) + ' alpha: ' + str(alpha) + ' steps: ' + str(steps))
            for i in range(tests):
                tour, solutions_count, performance = self.tsp_evo(k, n, alpha, steps)
                dist = round(self.calculate_total_distance(tour), 5)
                dists.append(dist)
                print('TEST ', i, ' |  Dist: ', dist, 
                    ' | Solutions Explored: ', solutions_count, 
                    '\n Tour: ', tour)

                content = '\nTEST ' + str(i) + ' |  Dist: ' + str(dist) +  ' | Solutions Explored: ' + str(solutions_count) + ' Tour: ' + str(tour) + '\nPerf Track: ' + str(performance)
                
                file.write(content)
        
            end = time.time()
            file.write('\nAvg Runtime: ' + str((end-start)/tests))
            file.write('\nAvg Dist: ' + str(sum(dists)/tests))


    def test_tsp_pop(self, tests, k, steps):
        print('k: ', k, ' step: ', steps)
        print('==============================')

        filename = 'test_pop_k' + str(k) + '_steps' + str(steps) + '.txt'
        start = time.time()
        dists = []
        with open(filename, 'w') as file:
            file.write('k: ' + str(k) + ' steps: ' + str(steps))
            for i in range(tests):
                tour, solutions_count, performance = self.tsp_pop(k, steps)
                dist = round(self.calculate_total_distance(tour), 5)
                dists.append(dist)
                print('TEST ', i, ' |  Dist: ', dist, 
                    ' | Solutions Explored: ', solutions_count, 
                    '\n Tour: ', tour)

                content = '\nTEST ' + str(i) + ' |  Dist: ' + str(dist) +  ' | Solutions Explored: ' + str(solutions_count) + ' Tour: ' + str(tour) + '\nPerf Track: ' + str(performance)
                
                file.write(content)

            end = time.time()
            file.write('\nAvg Runtime: ' + str((end-start)/tests))
            file.write('\nAvg Dist: ' + str(sum(dists)/tests))


    def run_test_sa(self):
        TESTS = 10
        T = 100 # initial temperature
        ALPHA = 0.99 # temp decay rate
        STEPS = 1000
        self.test_tsp_sa(TESTS,T,ALPHA,STEPS)

    def run_test_evo(self):
        TESTS = 10
        k = 8 # size of population
        n = 2 # number of swaps to make in mutation
        alpha = 8 # scaling factor for exaggerating probability of picking better solutions
        STEPS = 1000
        self.test_tsp_evo(TESTS,k,n,alpha,STEPS)

    def run_test_pop(self):
        TESTS = 10
        k = 4 # beam width
        STEPS = 1000
        self.test_tsp_pop(TESTS,k,STEPS)

    # Create a graph from a given number of cities
    def init_tsp(self):
        self.cities = []
        self.weights = {}
        #self.generate_random_tsp(int(num_cities_entry.get()))
        self.generate_from_file('hw2\hw2.csv')
        G = self.display_tsp_graph()
        ax.clear()
        ax.set_title('TSP')
        pos = nx.get_node_attributes(G, 'pos')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, ax=ax, node_size=300, node_color='lightblue', edge_color='grey', style='dashed')
        #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size = 6, alpha = 0.9, ax=ax)
        canvas.draw()

    # Update the graph when the "Solve" button is clicked
    def solve_tsp(self):
        # Solve tsp
        #tour, solutions_count, performance = self.tsp_brute_force() # returns tuple of city numbers for best tour
        method = str(solver_entry.get())
        if method == "sa":
            tour, solutions_count, performance = self.tsp_sa(100,0.9,1000)
        elif method == "evo":
            tour, solutions_count, performance = self.tsp_evo(16, 2, 8, 1000)
        elif method == "pop":
            tour, solutions_count, performance = self.tsp_pop(2,1000)
        

        # Generate directed graph with solution
        sol = self.display_tsp_graph(tour)
        
        edges = []
        for i in range(len(tour)-1):
            a = tour[i]
            b = tour[i+1]
            edges.append((a,b))

        # Do plotting
        pos = nx.get_node_attributes(sol, 'pos')
        labels = nx.get_edge_attributes(sol, 'weight')
        nx.draw_networkx_edges(sol, pos, edgelist = edges, edge_color = 'red', width = 2.0, ax=ax)
        #nx.draw_networkx_edge_labels(sol, pos, edge_labels=labels, font_size = 6, alpha = 0.9, ax=ax)
        
        tour_distance = self.calculate_total_distance(tour)
        total_solutions = 6.204484e+23/2 # calculated by hand using: (n-1)! permutations (2 directions for each sol)
        percent = (solutions_count/total_solutions) * 100

        solutions_label.config(text=f'Solutions Found: {solutions_count}')
        result_label.config(text=f'Total Distance: {tour_distance:.2f}')
        percent_label.config(text=f'Percent Explored: {percent:.10f}')
               
        canvas.draw()

        print('Tour: ', tour, 'Dist: ', tour_distance)

tsp = Graph_TSP_Solver()


# Create the main GUI window
root = tk.Tk()
root.title('Traveling Salesman Problem')

# Create and place widgets in the GUI
frame = tk.Frame(root)
frame.pack(pady=5)

init_button = tk.Button(frame, text='Init', command=tsp.init_tsp)
init_button.pack(side=tk.LEFT)

solver_label = tk.Label(frame, text='Enter Solver (sa, evo, pop)')
solver_label.pack(side=tk.LEFT)

solver_entry = tk.Entry(frame)
solver_entry.pack(side=tk.LEFT)

solve_button = tk.Button(frame, text='Solve', command=tsp.solve_tsp)
solve_button.pack(side=tk.LEFT)

test_sa_button = tk.Button(frame, text='SA Tests', command=tsp.run_test_sa)
test_sa_button.pack(side=tk.TOP)

test_evo_button = tk.Button(frame, text='Evo Tests', command=tsp.run_test_evo)
test_evo_button.pack(side=tk.TOP)

test_pop_button = tk.Button(frame, text='Pop Tests', command=tsp.run_test_pop)
test_pop_button.pack(side=tk.TOP)

result_label = tk.Label(root, text='')
result_label.pack(side=tk.TOP)

solutions_label = tk.Label(root, text='')
solutions_label.pack(side=tk.TOP)

percent_label = tk.Label(root, text='')
percent_label.pack(side=tk.TOP)


fig, ax = plt.subplots(figsize=(12, 9))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

root.mainloop()
