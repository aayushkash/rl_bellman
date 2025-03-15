# GridWorld Value Iteration

This project implements the Value Iteration algorithm for a 4x4 GridWorld environment. The agent starts in the top-left corner (state 0) and aims to reach the bottom-right corner (state 15).

## Problem Description

- **Environment**: 4x4 grid
- **Start State**: Top-left corner (0)
- **Goal State**: Bottom-right corner (15)
- **Actions**: Up, Down, Left, Right (equal probability 0.25)
- **Rewards**: -1 for each move, 0 for reaching the goal
- **Discount Factor (γ)**: 2
- **Convergence Threshold**: 1e-4

## Implementation Details

The implementation consists of three main functions:

1. `create_gridworld(size=4)`: Initializes the GridWorld environment with rewards
2. `get_next_state(state, action, size)`: Calculates the next state given current state and action
3. `value_iteration(size=4, gamma=2, theta=1e-4)`: Implements the Value Iteration algorithm

The Bellman equation is used iteratively to compute the optimal value function:

V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]

where:
- V(s) is the value of state s
- P(s'|s,a) is the transition probability (0.25 for all actions)
- R(s,a,s') is the reward (-1 for all moves except goal state)
- γ is the discount factor
- s' is the next state



## Output

The program outputs a 4x4 matrix representing the final value function for each state in the grid. Higher values indicate states that are closer to the goal state.
Final Value Function:

[[-0.4921875 -0.484375  -0.46875   -0.4375   ]
 [-0.484375  -0.46875   -0.4375    -0.375    ]
 [-0.46875   -0.4375    -0.375     -0.25     ]
 [-0.4375    -0.375     -0.25       0.       ]]



## Usage
- python
- from gridworld_value_iteration import value_iteration
- Run value iteration with default parameters
- V_final = value_iteration()
- print("\nFinal Value Function:")
- print(V_final)


- Or with custom parameters
- V_final = value_iteration(size=4, gamma=2, theta=1e-4)

## Requirements

- Python 3.x
- NumPy

## Installation

bash
pip install numpy

## File Structure
.
├── README.md
└── gridworld_value_iteration.py


## Notes

- The agent cannot move outside the grid boundaries
- The terminal state (bottom-right corner) maintains a value of 0
- The algorithm continues until the maximum change in values across all states is less than the convergence threshold
