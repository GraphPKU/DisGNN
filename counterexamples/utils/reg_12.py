
def generate_reg_12(rel_len: float) -> list:
    '''
        To generate a regular dodecahedron in XYZ coordinate system: 12 surfaces, 20 vertices
    '''
    u = 1
    s = (1 + 5 ** 0.5) / 2
    sr = 1 / s
    
    u, s, sr = u * rel_len, s * rel_len, sr * rel_len
    
    reg_12 = [
        [0, s, sr], [0, -s, sr], [0, -s, -sr], [0, s, -sr], 
        [sr, 0, s], [-sr, 0, s], [-sr, 0, -s], [sr, 0, -s], 
        [s, sr, 0], [-s, sr, 0], [-s, -sr, 0], [s, -sr, 0], 
        [u, u, u], [u, u, -u], [u, -u, u], [-u, u, u], 
        [-u, u, -u], [-u, -u, u], [u, -u, -u], [-u, -u, -u]
    ]
    
    return reg_12


'''
    Sample_policys: Contains all the sample policies that form counterexamples
        List of tuples, each tuple contains 3 elements:
            1. the number of points in the pair
            2. the sample policy of the left point set
            3. the sample policy of the right point set
'''

sample_policys_12 = []

# 6 points pair
policy_L = [
    1, 1, 1, 1, 
    0, 0, 0, 0,
    1, 0, 1, 0,
    0, 0, 0, 0,
    0, 0, 0, 0, 
]
policy_R = [
    1, 1, 1, 1, 
    1, 0, 1, 0, 
    0, 0, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 0, 0, 
]

sample_policys_12.append((policy_L, policy_R))

# 14 points pair
policy_L = [
    0, 0, 0, 0, 
    1, 1, 1, 1,
    0, 1, 0, 1,
    1, 1, 1, 1,
    1, 1, 1, 1, 
]
policy_R = [
    0, 0, 0, 0, 
    0, 1, 0, 1, 
    1, 1, 1, 1, 
    1, 1, 1, 1, 
    1, 1, 1, 1, 
]

sample_policys_12.append((policy_L, policy_R))

# 8 points pair
policy_L = [
    1, 1, 1, 1,
    1, 0, 1, 0,
    0, 0, 0, 0, 
    1, 0, 0, 0, 
    0, 0, 0, 1,
]
policy_R = [
    1, 1, 1, 1,
    0, 0, 0, 0,
    1, 0, 1, 0,
    1, 0, 0, 0,
    0, 0, 0, 1,
]

sample_policys_12.append((policy_L, policy_R))

# 12 points pair
policy_L = [
    0, 0, 0, 0,
    0, 1, 0, 1,
    1, 1, 1, 1, 
    0, 1, 1, 1, 
    1, 1, 1, 0,
]
policy_R = [
    0, 0, 0, 0,
    1, 1, 1, 1,
    0, 1, 0, 1,
    0, 1, 1, 1,
    1, 1, 1, 0,
]

sample_policys_12.append((policy_L, policy_R))

# first 10 points pair (the complementary pair is the same as the original pair)
policy_L = [
    0, 0, 0, 0, 
    0, 1, 0, 1, 
    1, 1, 1, 1, 
    0, 1, 0, 1, 
    0, 1, 1, 0,
]
policy_R = [
    0, 0, 0, 0, 
    1, 1, 1, 1,
    0, 1, 0, 1, 
    0, 0, 1, 1, 
    1, 0, 1, 0,
]

sample_policys_12.append((policy_L, policy_R))

# second 10 points pair (the complementary pair is the same as the original pair)
policy_L = [
    1, 1, 1, 1, 
    1, 0, 1, 0, 
    1, 0, 1, 0,
    0, 1, 0, 0,
    0, 1, 0, 0
]
policy_R = [
    1, 1, 1, 1,
    1, 0, 1, 0,
    1, 0, 1, 0,
    0, 0, 1, 0,
    1, 0, 0, 0
]

sample_policys_12.append((policy_L, policy_R))
