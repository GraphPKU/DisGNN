import networkx as nx
from builtins import str
from typing import Optional, Tuple, List

def color_aggr(G: nx.Graph):
    color_texts = {}
    label_list = nx.get_node_attributes(G,'label')
    for u in G.nodes():  
        color_text = []
        
        # multiset
        adj_nodes = G.adj[u]   
        for adj_node, attr in adj_nodes.items():
            label_adj = label_list[adj_node]
            tup = (label_adj, round(attr['weight'], 3))
            color_text.append(tup)
        color_text.sort()
        
        # self
        label_u = label_list[u]
        color_text.insert(0,label_u)
        
        color_texts[u] = str(color_text)
        
    return color_texts
        


def color_reassignment(color_text_l: dict, color_text_r: dict, Gl: nx.Graph, Gr: nx.Graph):
    color_idx = 1 # always relabel from 1
    
    # get the map from color text to new color index
    new_color = {}
    for color_text in color_text_l.values():
        if color_text not in new_color.keys():
            new_color[color_text] = color_idx
            color_idx += 1
    for color_text in color_text_r.values():
        if color_text not in new_color.keys():
            new_color[color_text] = color_idx
            color_idx += 1

    # relabel
    new_label_list_l, new_label_list_r = {}, {}
    for u in Gl.nodes():
        color_text = color_text_l[u]
        new_label_list_l[u] = new_color[color_text]
    nx.set_node_attributes(Gl, new_label_list_l, 'label')
    for u in Gr.nodes():
        color_text = color_text_r[u]
        new_label_list_r[u] = new_color[color_text]
    nx.set_node_attributes(Gr, new_label_list_r, 'label')
    
    return new_label_list_l, new_label_list_r
        


def CR_E(Gl: nx.Graph, Gr: nx.Graph, verbose: bool = False) -> Tuple[bool, Optional[dict], Optional[dict], int]:
    if verbose:
        print("-----------Beginning CR_E-----------")
    iteration = 0
    
    label_list_l, label_list_r = nx.get_node_attributes(Gl, 'label'), nx.get_node_attributes(Gr, 'label')
    if verbose:
        print("-----------Initial Labels-----------")
        print("Graph L:", label_list_l)
        print("Graph R:", label_list_r)
    while True:  
        # color refinement
        color_text_l, color_text_r = color_aggr(Gl), color_aggr(Gr) 
        new_label_list_l, new_label_list_r = color_reassignment(color_text_l,color_text_r,Gl,Gr)
        iteration += 1
        
        if new_label_list_l == label_list_l and new_label_list_r == label_list_r:
            break
        
        if verbose:
            print(f"-----------Iteration {iteration}-----------")
            print("Graph L:", new_label_list_l)
            print("Graph R:", new_label_list_r)
        
        result_l = sorted(map(int, new_label_list_l.values()))
        result_r = sorted(map(int, new_label_list_r.values()))
        
        if result_l != result_r:
            if verbose:
                print("-----------Ending CR-E-----------")
            return True, new_label_list_l, new_label_list_r, iteration
        
        
        label_list_l, label_list_r = new_label_list_l, new_label_list_r
        
    if verbose:
        print("-----------Ending CR-E-----------")
    return False, new_label_list_l, new_label_list_r, iteration


def CR_E_once(Gl: nx.Graph, Gr: nx.Graph, verbose: bool = False) -> Tuple[bool, bool]:
    if verbose:
        print("-----------Beginning CR_E Once-----------")
    
    label_list_l, label_list_r = nx.get_node_attributes(Gl, 'label'), nx.get_node_attributes(Gr, 'label')
    color_kind_num_l_before, color_kind_num_r_before = len(set(label_list_l.values())), len(set(label_list_r.values()))
    if verbose:
        print("-----------Labels Before-----------")
        print("Graph L:", label_list_l)
        print("Graph R:", label_list_r)
        print(f"-----------Color Kind Number Before-----------")
        print("Graph L:", color_kind_num_l_before)
        print("Graph R:", color_kind_num_r_before)
        
    color_text_l, color_text_r = color_aggr(Gl), color_aggr(Gr) 
    new_label_list_l, new_label_list_r = color_reassignment(color_text_l,color_text_r,Gl,Gr)
    color_kind_num_r_after, color_kind_num_l_after = len(set(new_label_list_l.values())), len(set(new_label_list_r.values()))

    if verbose:
        print(f"-----------Labels After-----------")
        print("Graph L:", new_label_list_l)
        print("Graph R:", new_label_list_r)
        print(f"-----------Color Kind Number After-----------")
        print("Graph L:", color_kind_num_l_after)
        print("Graph R:", color_kind_num_r_after)
    
    result_l = sorted(map(int, new_label_list_l.values()))
    result_r = sorted(map(int, new_label_list_r.values()))
    
    both_stable = (color_kind_num_l_before == color_kind_num_l_after and color_kind_num_r_before == color_kind_num_r_after)
    distinguishable = result_l != result_r
    

        
    if verbose:
        print("-----------Ending CR-E-----------")
    return distinguishable, both_stable