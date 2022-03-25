import importlib

def rel():
    import fp_utils
    import behavioral_data
    import fiber_data
    import analysis
    importlib.reload(behavioral_data)
    importlib.reload(fiber_data)
    importlib.reload(analysis)
    importlib.reload(fp_utils)