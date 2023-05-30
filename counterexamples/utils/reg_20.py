def generate_reg_20(rel_size: float) -> list:
    '''
        To generate a regular icosohedron in XYZ coordinate system: 20 surfaces, 12 vertices
    '''
    
    m = (5-(5**0.5))**0.5 * rel_size
    n = (5+(5**0.5))**0.5 * rel_size
    
    reg_20 = [
        [m, 0, n], [m, 0, -n], [-m, 0, -n], [-m, 0, n], 
        [0, n, m], [0, -n, m], [0, -n, -m], [0, n, -m], 
        [n, m, 0], [-n, m, 0], [-n, -m, 0], [n, -m, 0]
    ]
    
    return reg_20


'''
    Sample_policys: Contains all the sample policies that form counterexamples
        List of tuples, each tuple contains 3 elements:
            1. the number of points in the pair
            2. the sample policy of the left point set
            3. the sample policy of the right point set
'''

sample_policys_20 = []

# 6 points pair (the complementary pair is the same as the original pair)
policy_L = [
    1, 0, 1, 0, 
    0, 1, 0, 1, 
    0, 1, 0, 1, 
]
policy_R = [
    0, 1, 0, 1, 
    1, 0, 1, 0, 
    1, 0, 1, 0
]

sample_policys_20.append((policy_L, policy_R))
