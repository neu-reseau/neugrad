from engine import *

class help:
    def list(data):
        if isinstance(data,List):
            return data
        else:
            return data.tolist()
    def array(data):
        if isinstance(data,np.ndarray):
            return data
        elif isinstance(data,NVal):
            return data.toarray()
        else:
            return np.array(data)
    def nval(data):
        if isinstance(data,NVal):
            return data
        else:
            return NVal(data)