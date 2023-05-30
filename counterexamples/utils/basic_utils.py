from typing import List, Tuple
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np  
from typing import Optional
from counterexamples.utils.CR_E import CR_E
from typing import List, Tuple, Dict

def get_complementary_policy(sample_policy: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
    '''
        To get the complementary policy of a given policy
    '''
    
    policy_L, policy_R = sample_policy
 
    com_policy_L = [1 if i == 0 else 0 for i in policy_L]
    com_policy_R = [1 if i == 0 else 0 for i in policy_R]
        
    return (com_policy_L, com_policy_R)

def get_all_policy(sample_policy: Tuple[List[int], List[int]]) -> Tuple[List[int], List[int]]:
    '''
        To get the all policy of a given policy
    '''
    
    policy_L, policy_R = sample_policy
    
    all_policy_L = [1 for i in policy_L]
    all_policy_R = [1 for i in policy_R]
    
    return (all_policy_L, all_policy_R)


def sample(reg: List, policy: Tuple[List[int], List[int]]) -> Tuple[np.array, np.array]:
    '''
        To sample a set of points from a given regular polyhedron with a given policy
        Output are two point clouds: np.array, shape = (n, 3)
    '''
    
    pL, pR = policy
    assert len(reg) == len(pL) == len(pR)
    
    ary_reg = np.array(reg)
    ary_pL = np.array(pL)
    ary_pR = np.array(pR)
    
    PCl, PCr = (ary_reg[ary_pL == 1], ary_reg[ary_pR == 1])
    
    return (PCl, PCr)


def PointCloud2nxGraph(PC: np.array, initial_labels: Optional[dict] = None) -> nx.Graph:
    '''
        To convert a point cloud to a distance graph.
        Input a point cloud: np.array, shape = (n, 3)
        Middle is a distance graph: np.array, shape = (n, n)
        Output is a networkx graph
    '''
    DG = np.sqrt(
            np.sum(
                (PC[:, np.newaxis, :] - PC[np.newaxis, :, :]) ** 2, 
                axis=2
                )
            )
    
    G = nx.from_numpy_matrix(DG, create_using=nx.Graph)
    
    # initial label
    if initial_labels is None:
        for i in range(len(G.nodes)):
            G.nodes[i]['label'] = 1      
    else:
        assert len(initial_labels.keys()) == len(G.nodes)
        nx.set_node_attributes(G, initial_labels, 'label')
    
    return G


def TestIsomorphism(Gl: np.array, Gr: np.array) -> bool:
    '''
        To test whether two distance graphs are isomorphic.
        
        Input two distance graphs: np.array, shape = (n, n)
    '''
    
    em = iso.categorical_edge_match('weight', 'weight')
    GM = iso.GraphMatcher(Gl, Gr, edge_match=em)
    
    return GM.is_isomorphic()


def visualize(labels_l_dict: Dict[int, int], labels_r_dict: Dict[int, int], PCl: np.array, PCr: np.array, com_PCl: np.array, com_PCr: np.array):
    '''
        To visualize the result of the algorithm
    '''
    import matplotlib.pyplot as plt
    rainbow_colors = {1: "#FF0000", 2: "#FFA500", 3: "#FFFF00", 4: "#00FF00", 5: "#007FFF", 6: "#0000FF", 7: "#8B00FF"}

    labels_l_list, labels_r_list = list(labels_l_dict.values()), list(labels_r_dict.values())
    color_list_l, color_list_r = [rainbow_colors[i] for i in labels_l_list], [rainbow_colors[i] for i in labels_r_list]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(com_PCl[:, 0], com_PCl[:, 1], com_PCl[:, 2], c="gray")
    ax1.scatter(PCl[:, 0], PCl[:, 1], PCl[:, 2], c=color_list_l)
    
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.scatter(com_PCr[:, 0], com_PCr[:, 1], com_PCr[:, 2], c="gray")
    ax2.scatter(PCr[:, 0], PCr[:, 1], PCr[:, 2], c=color_list_r)
    plt.show()
    
def get_label_histogram(label_dict: dict) -> List[int]:
    '''
        To convert a label dictionary to a label list
    '''
    label_list = list(label_dict.values())
    label_l_count = {}
    for label in label_list:
        label_l_count[str(label)] = label_l_count.get(str(label), 0) + 1
    label_l_count = sorted(label_l_count.items(), key=lambda x: x[0])
    return label_l_count