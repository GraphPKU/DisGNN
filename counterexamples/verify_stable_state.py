import sys
sys.path.append(".")
from argparse import ArgumentParser
from counterexamples.utils.reg_12 import generate_reg_12, sample_policys_12
from counterexamples.utils.reg_20 import generate_reg_20, sample_policys_20
from counterexamples.utils.basic_utils import sample, TestIsomorphism, PointCloud2nxGraph, get_complementary_policy, get_all_policy
from counterexamples.utils.CR_E import CR_E, CR_E_once
import random

# Args
parser = ArgumentParser()
parser.add_argument('--reg', choices=["12", "20"], default="12")
parser.add_argument('--verbose', action="store_true")
args = parser.parse_args()
verbose = args.verbose

if args.reg == "12":
    generate_reg = generate_reg_12
    sample_policys = sample_policys_12
    reg_name = "regular dodecahedron"
else:
    generate_reg = generate_reg_20
    sample_policys = sample_policys_20
    reg_name = "regular icosahedron"
    
    
    
if __name__ == "__main__":

    rel_len = random.uniform(0.1, 10)
    print(f"The random rel_len = {rel_len}.")
    reg = generate_reg(rel_len)

    for policy in sample_policys:
        node_num = sum(policy[0])
        print(f"\nTesting the {node_num}-points pair sampled on {reg_name}...")
        assert sum(policy[0]) == sum(policy[1])
        ori_policy = policy
        # get the complementary policy and the all policy
        com_policy = get_complementary_policy(ori_policy)
        all_policy = get_all_policy(ori_policy)
        
        # get the final state of original policy and complementary policy
        PCl_ori, PCr_ori = sample(reg, ori_policy)
        Gl_ori, Gr_ori = PointCloud2nxGraph(PCl_ori, initial_labels=None), PointCloud2nxGraph(PCr_ori, initial_labels=None)
        distinguisable_ori, final_state_l_ori, final_state_r_ori, _ = CR_E(Gl=Gl_ori, Gr=Gr_ori, verbose=verbose)
        
        PCl_com, PCr_com = sample(reg, com_policy)
        Gl_com, Gr_com = PointCloud2nxGraph(PCl_com, initial_labels=None), PointCloud2nxGraph(PCr_com, initial_labels=None)
        distinguisable_com, final_state_l_com, final_state_r_com, _ = CR_E(Gl=Gl_com, Gr=Gr_com, verbose=verbose)
        
        assert distinguisable_ori == False and distinguisable_com == False
        
        # construct the stable state for all policy
        all_node_num = sum(all_policy[0])
        stable_state_l_all, stable_state_r_all = {}, {}
        for i in range(all_node_num):
            if com_policy[0][i] == 1:
                node_idx_in_com = sum([1 for j in range(i) if com_policy[0][j] == 1])
                stable_state_l_all[i] = final_state_l_com[node_idx_in_com]
            else:
                node_idx_in_ori = sum([1 for j in range(i) if ori_policy[0][j] == 1])
                stable_state_l_all[i] = final_state_l_ori[node_idx_in_ori] + all_node_num # ensure that nodes in different part have different labels
                
            if com_policy[1][i] == 1:
                node_idx_in_com = sum([1 for j in range(i) if com_policy[1][j] == 1])
                stable_state_r_all[i] = final_state_r_com[node_idx_in_com]
            else:
                node_idx_in_ori = sum([1 for j in range(i) if ori_policy[1][j] == 1])
                stable_state_r_all[i] = final_state_r_ori[node_idx_in_ori] + all_node_num
        
        # construct the nx.Graph for all policy with initial labels (i.e. stable state)
        PCl_all, PCr_all = sample(reg, all_policy)
        Gl_all, Gr_all = PointCloud2nxGraph(PCl_all, initial_labels=stable_state_l_all), PointCloud2nxGraph(PCr_all, initial_labels=stable_state_r_all)
        

        # Test if CR-E can distinguish them
        distinguisable, both_stable = CR_E_once(Gl=Gl_all, Gr=Gr_all, verbose=verbose)
        
        # If they are both stable, and CR-E cannot distinguish them, then the lemma is verified.
        if (not distinguisable) and both_stable:
            print(f"Lemma verified for {reg_name} and original policy with node num {sum(ori_policy[0])}")
        else:
            print(f"Lemma not verified for {reg_name} and original policy with node num {sum(ori_policy[0])}")
        