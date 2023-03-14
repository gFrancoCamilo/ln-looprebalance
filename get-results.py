from topology import *
from results import *
from pcn import *
from tqdm import tqdm
import click

@click.group(chain=True, invoke_without_command=True)
def cli():
    pass

@cli.command(name='success-ratio', help='Gets payment success ratio')
@click.option('-b','--balance', default='ln', type=click.Choice (['ln','99-1','half'],
             case_sensitive=False), help = 'Define initial balance used in the simulation')
@click.option('-b_parameter','--balance-parameter', default=0.1, type=float,
            help='Defines the percentage of highest degree node that will be balanced 50-50')
@click.option('-n_payments','--number_payments', type=int, default=100,
             help='Number of payments in the simualtion')
@click.option('-t','--topology', default='lightning',
             type=click.Choice (['scale-free','watts-strogatz','barabasi-albert', 'lightning'], case_sensitive=False),
             help = 'Graph topology used in the simulation')
@click.option('-n','--nodes', type=int, default=10, help='Number of nodes in the topology. Only used if topology is not lightning.')
@click.option('--alpha', default=0.5, help='Alpha parameter for scale-free topology. Only used with scale-free topology.')
@click.option('--beta', default=0.00001, help='Beta parameter for scale-free topology. Only used with scale-free topology.')
@click.option('--gamma', default=0.49999, help='Gamma parameter for scale-free topology. Only used with scale-free topology.')
@click.option('-k', default=2, help='K parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-p', default=0.1, help='P parameter for Watts-Strogatz graph. Only used with Watts-Strogatz topology.')
@click.option('-m', default=2, help='M parameter for Barabasi-Albert graph. Only used with Watts-Strogatz topology.')
@click.option('-d', '--date', default='jul 2022', type=click.Choice(['jul 2021', 'jan 2022', 'jul 2022'], case_sensitive=False),
            help='Date of lighting snapshot to be used in the simulation. Only used with lightning topology.')
@click.option('-pay','--payment_method', default='ripple', type=click.Choice(['ripple','credit-card'], case_sensitive=False),
            help='Dataset used to simulate payment in the network.')
def simulate_success_ratio (balance, balance_parameter, number_payments, topology, nodes, alpha, beta, gamma, k, p, m, date, payment_method):
    results = []
    if topology == 'lightning':
        Graph = graph_names(date)
        if balance != 'ln':
            Graph = set_balance(Graph, balance)
        else:
            Graph = set_balance_ln(Graph, balance_parameter)
        payment_dataset = get_payment_dataset(payment_method)
        payment_list = choose_payments(payment_dataset, number_payments)
        payment_dict = get_payments_ln(Graph, payment_list)
        for i in range(10):
            results.append(get_success_ratio(Graph, payment_dict))


if __name__ == '__main__':
    cli()