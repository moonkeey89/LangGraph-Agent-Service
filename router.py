def decide_next_step(state):


    if state.status=="success":

        return "response"


    elif state.status=="failed":

        return "planner"


    else:

        return "planner"