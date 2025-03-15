import numpy as np

def create_gridworld(size=4):
    """Initialize the GridWorld environment."""
    n_states = size * size
    rewards = np.full((size, size), -1.0)
    rewards[size-1, size-1] = 0.0  # Terminal state
    return size, rewards

def get_next_state(state, action, size):
    """Get next state given current state and action."""
    row = state // size
    col = state % size
    
    # Initialize next_row and next_col with current position
    next_row = row
    next_col = col
    
    if action == 0:  # up
        next_row = max(0, row - 1)
    elif action == 1:  # down
        next_row = min(size - 1, row + 1)
    elif action == 2:  # left
        next_col = max(0, col - 1)
    else:  # right
        next_col = min(size - 1, col + 1)
    
    return next_row * size + next_col

def value_iteration(size=4, gamma=2, theta=1e-4):
    """Perform value iteration for the GridWorld."""
    size, rewards = create_gridworld(size)
    n_states = size * size
    n_actions = 4  # up, down, left, right
    V = np.zeros((size, size))
    
    while True:
        delta = 0
        V_new = V.copy()
        
        # For each state
        for row in range(size):
            for col in range(size):
                if row == size-1 and col == size-1:  # Skip terminal state
                    continue
                
                state = row * size + col
                v = float('-inf')
                
                # For each action
                for action in range(n_actions):
                    next_state = get_next_state(state, action, size)
                    next_row = next_state // size
                    next_col = next_state % size
                    
                    # Equal probability for all actions (0.25)
                    v_action = 0.25 * (rewards[row, col] + gamma * V[next_row, next_col])
                    v = max(v, v_action)
                
                V_new[row, col] = v
                delta = max(delta, abs(V_new[row, col] - V[row, col]))
        
        V = V_new
        if delta < theta:
            break
    
    return V

# Run the algorithm and print results
V_final = value_iteration()
print("\nFinal Value Function:")
print(V_final) 