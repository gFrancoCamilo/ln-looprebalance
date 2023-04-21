from rebalance import *
from payments import *
from results import *

def init_rebalance_test (Graph: nx.DiGraph, payment_dict, threshold: float = 0.3, delay: int = 5):
    try:
        edges = [('038fe1bd966b5cb0545963490c631eaa1924e2c4c0ea4e7dcb5d4582a1e7f2f1a5', '02f3069a342ae2883a6f29e275f06f28a56a6ea2e2d96f5888a3266444dcf542b6'),
                ('02ec921faa245b5813385e04a51a6f0e2d99a570ff820bc42b6d1048c278f21216', '035e4ff418fc8b5554c5d9eea66396c227bd429a3251c8cbc711002ba215bfc226'),
                ('03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f', '03dc686001f9b1ff700dfb8917df70268e1919433a535e1fb0767c19223509ab57'),
                ('03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f', '02634002b0b3469695a3cd5a29e363071a09374c7753b3aea34d1fc297226b4dba'),
                ('03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f', '03683e95c542d5aae980b2cb346b6e5ff2416222b6962cf3f23dfbf4f93b409643'),
                ('03c8dfbf829eaeb0b6dab099d87fdf7f8faceb0c1b935cd243e8c1fb5af71361cf', '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f'),
                ('026165850492521f4ac8abd9bd8088123446d126f648ca35e60f88177dc149ceb2', '03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f'),
                ('03c8dfbf829eaeb0b6dab099d87fdf7f8faceb0c1b935cd243e8c1fb5af71361cf', '022755c3ff4e5a1d71f573cda4b315887fc00a9e5c9ea9a847d939f3e517e69a70')]
        
        event = threading.Event()
        for edge in edges:
            init_rebalance (Graph, event, edge, 'pickhardt', threshold, delay)
        
        get_success_ratio(Graph, payment_dict)
        event.set()
        
    except Exception as e:
        event.set()
        print (e)
Graph = graph_names ('jul 2022')
Graph = validate_graph(Graph)

Graph = set_balance_ln(Graph, 0.01)
#Graph = set_balance(Graph, 'half')
payment_dataset = get_payment_dataset('ripple')
payment_list = choose_payments(payment_dataset, 100)
payment_dict = get_payments_ln(Graph, payment_list)
init_rebalance_test(Graph, payment_dict)
