from topology import *
import time

def generate_timestamps_test():
    """
    Tests the generate_timestamps function in topology.py. Prints
    the list of filenames and dates in DD-MM-YYYY format on the
    screen
    """
    try:
        filenames, dates = generate_timestamps()
        print("Filename: " + str(filenames))
        print("Dates: " + str(dates))
        return filenames
    except:
        raise Exception ('Failed executing generate_timestamps function in topology.py')

def create_graph_test(filenames, test_all = 1):
    """
    Tests the create_graph function from topology.py. As creating
    every graph snapshot is costly, we set an argument that allows
    testing with the most recent snapshot. Thus, if the argument
    test_all is set to one, the function will test all snapshots.
    Otherwise, the function will only create the most recent one.
    """
    try:
        if test_all == 1:
            for file in filenames:
                Graph = create_graph(file)
                print("Created graph from file " + str(file))
        else:
            Graph = create_graph(filenames[len(filenames) - 1])
            print("Created graph from file " + str(filenames[len(filenames) - 1]))    
        return Graph
    except:
        raise Exception ('Failed executing create_graph function in topology.py')

def count_node_triangles_test(Graph):
    """
    Tests the count_node_triangle function from topology.py. 
    As the function requires a node to test, we use the ACINQ
    node given that it is a strongly connected central node.
    """
    try:
        number_triangles = count_node_triangles(Graph, "03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f")
        print("Number of triangles: " + str(number_triangles))
    except:
        raise Exception ('Failed executing count_node_triangles function in topology.py')

def nodes_betweenness_centrality_test (Graph, processes):
    """
    Tests the nodes_betweenness_centrality function from topology.py.
    As the function uses multithreading, the test function accepts
    the number of threads an argument to be tested
    """
    try:
        bc = nodes_betweenness_centrality(Graph, processes)
        print("Betweenness centrality: " + str(bc))
    except:
        raise Exception ('Failed executing nodes_betweenness_centrality function in topology.py')

def nodes_closeness_centrality_test (Graph):
    """
    Tests the nodes_closeness_centrality function from topology.py.
    """
    try:
        print(nodes_closeness_centrality(Graph))
    except:
        raise Exception ('Failed executing nodes_closeness_centrality function in topology.py')

def triadic_census_test(Graph):
    """
    Tests the triadic_census function from topology.py.
    """
    try:
        triadic_class = triadic_census(Graph)
        print("Triadic Census: " + str(triadic_class))
    except:
        raise Exception ('Failed executing triadic_census function in topology.py')


filenames = generate_timestamps_test()
Graph = create_graph_test(filenames, 0)
count_node_triangles_test(Graph)
start = time.time()
nodes_betweenness_centrality_test(Graph, 15)
end = time.time()
print("Time: " + str(end-start))
#nodes_closeness_centrality_test(Graph)
#triadic_census_test(Graph)
