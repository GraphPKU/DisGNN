import sys
sys.path.append(".")
from argparse import ArgumentParser
from counterexamples.utils.basic_utils import sample, TestIsomorphism, PointCloud2nxGraph
from counterexamples.utils.CR_E import CR_E
import random

# Args
parser = ArgumentParser()
parser.add_argument('--reg', choices=["12", "20", "cube_8", "cube_cube"], default="12")
parser.add_argument('--verbose', action="store_true")
parser.add_argument('--visualize', action="store_true")
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
elif args.reg == "cube_8":
    from counterexamples.utils.cube_reg_8 import generate_cube_plus_reg_8, sample_policys_cube_plus_reg_8
    generate_reg = generate_cube_plus_reg_8
    sample_policys = sample_policys_cube_plus_reg_8
    reg_name = "cube plus regular octahedron"
elif args.reg == "cube_cube":
    from counterexamples.utils.cube_cube import generate_cube_plus_cube, sample_policys_cube_plus_cube
    generate_reg = generate_cube_plus_cube
    sample_policys = sample_policys_cube_plus_cube
    reg_name = "cube plus cube"

    
if __name__ == "__main__":

    rel_len = random.uniform(0.1, 10)
    print(f"The random rel_len = {rel_len}.")
    reg = generate_reg(rel_len)

    for policy in sample_policys:
        assert sum(policy[0]) == sum(policy[1])
        node_num = sum(policy[0])
        print(f"\nTesting the {node_num}-points pair sampled from {reg_name}...")
        
        PCl, PCr = sample(reg, policy)
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
            
            # If visualize is True, visualize the counterexample
            if args.visualize:
                from counterexamples.utils.basic_utils import visualize, get_complementary_policy
                com_PCl, com_PCr = sample(reg, get_complementary_policy(policy))
                visualize(labels_l_dict, labels_r_dict, PCl, PCr, com_PCl, com_PCr)