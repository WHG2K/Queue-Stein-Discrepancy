import numpy as np

def sample_on_grid(times, Q, t_grid):
    """
    Given event times 'times' and right-constant levels 'Q', return Q(t_grid).
    Assumes times[0]=0 and len(Q)==len(times), and Q is piecewise-constant
    right-continuous with jumps at 'times'.
    """
    idx = np.searchsorted(times, t_grid, side='right') - 1
    idx = np.clip(idx, 0, len(Q)-1)
    return np.array([Q[i] for i in idx], dtype=int)




if __name__ == "__main__":
    # Example usage
    times = np.array([0, 2, 5, 7, 10])
    Q = [(0,0), (1,0), (1,1), (0,1), (0,0)]
    t_grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    sampled_Q = sample_on_grid(times, Q, t_grid)
    print("Sampled Q on grid:", sampled_Q)