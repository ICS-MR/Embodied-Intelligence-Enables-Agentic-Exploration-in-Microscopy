from fastapi import Request


def get_runtime_manager(request: Request):
    return request.app.state.runtime_manager

