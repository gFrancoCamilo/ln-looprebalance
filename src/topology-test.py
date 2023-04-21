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

def graph_names_test (option: str):
    """
    Tests the graph_names_test function from topology.py. The function
    tests all possible options for ln graph available. It also asserts
    that the returned Graph contains only one strongly connected component.
    """
    try:
        options = ["jul 2021", "jan 2022", "jul 2022"]
        for option in options:
            Graph = graph_names(option)
        assert nx.is_strongly_connected(Graph) == True
        return Graph
    except:
        raise Exception ('Failed executing graph_names function in topology.py')

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
            Graph = create_graph(filenames[0])
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

def edges_betweenness_centrality_test (Graph, processes):
    """
    Tests the nodes_betweenness_centrality function from topology.py.
    As the function uses multithreading, the test function accepts
    the number of threads an argument to be tested
    """
    try:
        bc = edges_betweenness_centrality(Graph, processes)
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

def get_k_most_centralized_nodes_test (Graph, k):
    """
    Tests the get_k_most_centralized_nodes function from topology.py
    """
    try:
        centralized = get_k_most_centralized_nodes(Graph, k)
        print(centralized)
    except:
        raise Exception ('Failed to get most centralized nodes')

def snowball_sample_test (Graph, node = None, size = 200):
    """
    snowball_sample_test tests snowball_sample from topology.py
    """
    try:
        Graph = snowball_sample(Graph, 'random', size)
        print("Number of nodes: " + str(Graph.number_of_nodes()))
        print("Number of edges: " + str(Graph.number_of_edges()))
    except:
        raise Exception ("Could not sample graph")

Graph = graph_names_test('jul 2022')
snowball_sample_test (Graph)
#filenames = generate_timestamps_test()
#Graph = create_graph_test(filenames, 0)
#get_k_most_centralized_nodes_test(Graph, 50)
#count_node_triangles_test(Graph)
#start = time.time()
#dic = nx.get_edge_attributes(Graph, "fee_base_msat")
#for edge in dic:
#    (u, v) = edge
#    Graph[u][v]["fee_base_msat"] = dic[edge] + 1
        
#edges_betweenness_centrality_test(Graph, 25)
#print(dic[('0268fb5ff483d584b81832b025b8ed122596d4b642171d71ab8b5893aa24eccece', '0383b3e460dfacc59c7b87dab8b5b72a29a481e195c85d7bfc55219ded164ad9ea')])
#(dic[('033d8656219478701227199cbd6f670335c8d408a92ae88b962c49d4dc0e83e025', '0348cb5eb33343108bf04930c416630af5795484791035911a330dfa6964823c7a')])
#print(dic[('03ff89e6e0062dd2a798b0a9570478bb3218f72fb32bfe929418796238f854ed80', '033d8656219478701227199cbd6f670335c8d408a92ae88b962c49d4dc0e83e025')])
#print(dic[('03e5a10bd9d700cfefd89dcfb423e3b8124895feae20be70a967db4443594106e7', '03945a8a49c380f5bfe39bddd083a74d2dc4d223fbb146c8ab8bab29b154150b7e')])
#print(dic[('0382feff4b67e914bae5103b341a8f4933664306c72123fcfbcc77875240894414', '038cb1f9d4eb94d7aee59cae45bd00727e69e83f45b63c2e4dad71112af789a862')])
#print(nx.edge_betweenness_centrality(Graph, weight='fee_base_msat'))
#end = time.time()
#print("Time: " + str(end-start))
#nodes_closeness_centrality_test(Graph)
#triadic_census_test(Graph)
