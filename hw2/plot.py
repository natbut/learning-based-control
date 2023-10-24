import matplotlib.pyplot as plt
import ast
import statistics as st

def process_test_perf(filename):
    test_perf_values = []
    running_test_perf = []
    
    print('Values for ' + filename)

    dists = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("TEST"):
                # A new "Test" is encountered
                if running_test_perf: # if content is stored
                    # If we have accumulated values, add them to the list
                    test_perf_values.append(running_test_perf)
                running_test_perf = [] #reset performance tracker
                back = line.split('Dist: ')[1].strip()
                dist = back.split(' | ')[0].strip()
                dists.append(float(dist))

            elif line.startswith("Perf Track"):
                # A new "Performance" value is encountered
                # start running test perf as rest of line
                running_test_perf = ast.literal_eval(line.split('Perf Track:')[-1].strip())
            
            elif line.startswith("Avg Runtime"):
                print(line)
            
            elif line.startswith("Avg Dist"):
                print(line)
                
            elif running_test_perf: #if tracking a test
                # If we are accumulating values
                test_perf = ast.literal_eval(line) # float(line.strip())
                running_test_perf.append(test_perf)

    # If there are accumulated values at the end of the file, add them
    if running_test_perf:
        test_perf_values.append(running_test_perf)

    if not test_perf_values:
        print("No 'Perf Track' values found in the file.")
        return
    
    print('Dist std dev: ', st.stdev(dists))
    
    print('============')
    return test_perf_values

def process_and_plot_test_perf(filename):
    test_perf_values = []
    running_test_perf = []
    

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("TEST"):
                # A new "Test" is encountered
                if running_test_perf: # if content is stored
                    # If we have accumulated values, add them to the list
                    test_perf_values.append(running_test_perf)
                running_test_perf = [] #reset performance tracker

            elif line.startswith("Perf Track"):
                # A new "Performance" value is encountered
                # start running test perf as rest of line
                running_test_perf = ast.literal_eval(line.split('Perf Track:')[-1].strip())
            
            elif line.startswith("Avg Runtime"):
                print(line)
            
            elif line.startswith("Avg Dist"):
                print(line)
                
            elif running_test_perf: #if tracking a test
                # If we are accumulating values
                test_perf = ast.literal_eval(line) # float(line.strip())
                running_test_perf.append(test_perf)

    # If there are accumulated values at the end of the file, add them
    if running_test_perf:
        test_perf_values.append(running_test_perf)

    if not test_perf_values:
        print("No 'Perf Track' values found in the file.")
        return

    # Plotting the 'Test Perf' values for each test
    for i, perf_list in enumerate(test_perf_values):
        plt.plot(perf_list, label=f'Test {i + 1}')

    plt.title("Performance Over Time")
    plt.xlabel("Step")
    plt.ylabel("Shortest Distance")
    plt.legend()
    plt.grid(True)
    plt.show()