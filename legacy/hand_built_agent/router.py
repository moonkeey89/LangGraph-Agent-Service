def entry_router(state):
    return state.next_step


def router(state):
    if state.status == "success":
        return "response"

    if state.status == "failed":
        if state.retry_count < 3:
            return "planner"

        return "response"

    return None
