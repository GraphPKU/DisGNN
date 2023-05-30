import sys
sys.path.append(".")
from argparse import ArgumentParser
from counterexamples.utils.basic_utils import sample, TestIsomorphism, PointCloud2nxGraph, get_complementary_policy, get_all_policy
from counterexamples.utils.CR_E import CR_E
import random
import numpy as np

# Args
parser = ArgumentParser()
parser.add_argument('--reg', choices=["12", "20"], default="12")
parser.add_argument('--verbose', action="store_true")
parser.add_argument('--num_of_layers', type=int, default=None)
args = parser.parse_args()
verbose = args.verbose

if args.reg == "12":
    from counterexamples.utils.reg_12 import generate_reg_12, sample_policys_12
    generate_reg = generate_reg_12
    sample_policys = sample_policys_12
    reg_name = "regular dodecahedron"
elif args.reg == "20":
    from counterexamples.utils.reg_20 import generate_reg_20, sample_policys_20
    generate_reg = generate_reg_20
    sample_policys = sample_policys_20
    reg_name = "regular icosahedron"

    
if __name__ == "__main__":
    if args.num_of_layers is None:
        number_of_layers = random.randint(2, 10)
    else:
        number_of_layers = args.num_of_layers
    layer_types = []
    while True:
        for _ in range(number_of_layers):
            layer_types.append(random.choice(["ori", "com", "all"]))
        if layer_types != ["all"] * number_of_layers:
            break

    rel_sizes = set()
    while True:
        rel_sizes.add(random.uniform(0.1, 10))
        if len(rel_sizes) == number_of_layers:
            break
    rel_sizes = list(rel_sizes)


    print(f"The random number_of_layers = {number_of_layers}")
    print(f"The random layer_types = {layer_types}")
    print(f"The random rel_sizes = {rel_sizes}")


    for policy in sample_policys:
        assert sum(policy[0]) == sum(policy[1])
        node_num = sum(policy[0])
        print(f"\nTesting the counterexample family based on the {node_num}-points graph sampled from {reg_name}...")
        PCl, PCr = np.zeros((0, 3)), np.zeros((0, 3))
        for i in range(number_of_layers):
            layer_type = layer_types[i]
            if layer_type == "ori":
                the_policy = policy
            elif layer_type == "com":
                the_policy = get_complementary_policy(policy)
            elif layer_type == "all":
                the_policy = get_all_policy(policy)
            
            rel_size = rel_sizes[i]
            the_reg = generate_reg(rel_size)
                
            the_PCl, the_PCr = sample(the_reg, the_policy)
            PCl, PCr = np.concatenate((PCl, the_PCl)), np.concatenate((PCr, the_PCr))

        
        
        Gl, Gr = PointCloud2nxGraph(PCl, initial_labels=None), PointCloud2nxGraph(PCr, initial_labels=None)
        
        # Test if two graphs are isomorphic
        is_isomorphic = TestIsomorphism(Gl, Gr)
        # right align 
        print(f"{' ' * 4}Isomorphic?{' ' * 14}{is_isomorphic}")

        # Test if CR-E can distinguish them
        distinguisable, labels_l_dict, labels_r_dict, _ = CR_E(Gl=Gl, Gr=Gr, verbose=verbose)
        print(f"{' ' * 4}CR-E Distinguishable?{' ' * 4}{distinguisable}")
        
        # If they are not isomorphic and CR-E can not distinguish them, then it is a counterexample
        is_counterexample = (not is_isomorphic) and (not distinguisable)
        print(f"{' ' * 4}Counterexample?{' ' * 10}{is_counterexample}")
        
        # If it is a counterexample, print the label histogram
        if is_counterexample:
            from counterexamples.utils.basic_utils import get_label_histogram
            hist_l, hist_r = get_label_histogram(labels_l_dict), get_label_histogram(labels_r_dict)
            print(f"{' ' * 4}Label Histogram:")
            print(f"{' ' * 8}Left: {hist_l}")
            print(f"{' ' * 8}Right: {hist_r}")
