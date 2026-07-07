from value import Value

topo = []
visited = set()

def build_topo(v: Value):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)