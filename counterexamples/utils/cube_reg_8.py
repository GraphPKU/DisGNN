from counterexamples.utils.basic_utils import get_complementary_policy

def generate_cube_plus_reg_8(rel_len: float) -> list:
    '''
        To generate a cube plus a regular octahedron in XYZ coordinate system.
    '''
    a = 1
    
    cube = [
        [a, a, a], [a, a, -a], [a, -a, a], [-a, a, a],
        [-a, a, -a], [-a, -a, a], [a, -a, -a], [-a, -a, -a]
    ]
    
    b = rel_len

    reg_8 = [
        [b, 0, 0], [-b, 0, 0], 
        [0, b, 0], [0, -b, 0], 
        [0, 0, b], [0, 0, -b]
    ]
    
    cube_plus_reg_8 = cube + reg_8
    
    return cube_plus_reg_8


'''
    Sample_policys: Contains all the sample policies that form counterexamples
        List of tuples, each tuple contains 3 elements:
            1. the number of points in the pair
            2. the sample policy of the left point set
            3. the sample policy of the right point set
'''

sample_policys_cube_plus_reg_8 = []

# 6 points pair
policy_L = [
    0, 1, 0, 1,
    0, 1, 1, 0,
    0, 0,
    1, 1,
    0, 0,
]

policy_R = [
    0, 1, 0, 1,
    0, 1, 1, 0,
    0, 0,
    0, 0,
    1, 1,
]

sample_policys_cube_plus_reg_8.append((policy_L, policy_R))

# 8 points pair
policy_L = [
    1, 0, 1, 0,
    1, 0, 0, 1,
    1, 1,
    0, 0,
    1, 1,
]

policy_R = [
    1, 0, 1, 0,
    1, 0, 0, 1,
    1, 1,
    1, 1,
    0, 0,
]

sample_policys_cube_plus_reg_8.append((policy_L, policy_R))