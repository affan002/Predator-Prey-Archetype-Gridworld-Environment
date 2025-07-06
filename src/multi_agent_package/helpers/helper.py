
def print_action(action):
    num_to_dir = {0: 'UP', 1: 'DOWN', 2:'RIGHT', 3:'LEFT', 4:'NOOP'}
    print(f"{num_to_dir[action]} ({action})")


def print_mdp_tuple(episode, state, action,input_tuple):
    mdp_dict = {'next_state' : input_tuple[0],
                'reward' : input_tuple[1],
                'done' : input_tuple[2], 
                'trunc': input_tuple[3], 
                'info' : input_tuple[4]}
    
    print(f" << episode --> {episode} >>")
    print(f"current state --> {state}")
    print_action(action)
    for key in mdp_dict.keys():
        print(f"{key} --> {mdp_dict[key]}")

    print("=====================================")


