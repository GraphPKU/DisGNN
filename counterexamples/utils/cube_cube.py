from counterexamples.utils.basic_utils import get_complementary_policy

def generate_cube_plus_cube(rel_len: float) -> list:
    '''
        To generate a cube plus a regular octahedron in XYZ coordinate system.
    '''
    a = 1
    
    cube1 = [
        [a, a, a], [a, a, -a], [a, -a, a], [-a, a, a],
        [-a, a, -a], [-a, -a, a], [a, -a, -a], [-a, -a, -a]
    ]
    
    b = rel_len
    c = 2 ** 0.5 * b

    cube2 = [
        [c, 0, b], [c, 0, -b], [-c, 0, b], [-c, 0, -b],
        [0, c, b], [0, c, -b], [0, -c, b], [0, -c, -b],
    ]
    
    cube_plus_cube = cube1 + cube2
    
    return cube_plus_cube


'''
    Sample_policys: Contains all the sample policies that form counterexamples
        List of tuples, each tuple contains 3 elements:
            1. the number of points in the pair
            2. the sample policy of the left point set
            3. the sample policy of the right point set
'''

sample_policys_cube_plus_cube = []

# 6 points pair
policy_L = [
    0, 1, 0, 1,
    0, 1, 1, 0,
    
    0, 0, 0, 0,
    1, 1, 1, 1,
]

policy_R = [
    0, 0, 1, 1,
    1, 0, 1, 0,

    0, 0, 0, 0,
    1, 1, 1, 1,
]

sample_policys_cube_plus_cube.append((policy_L, policy_R))

