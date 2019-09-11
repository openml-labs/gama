

class Emission(object):

    def __init__(self, primitive=None, parameters=None):
        self.primitive = primitive
        self.parameters = parameters

    def __eq__(self, other):
        return self.primitive.identifier == other.primitive.identifier and self.parameters == other.parameters

    def __hash__(self):
        if self.parameters is None:
            pkey = None
        else:
            pkey = tuple((k, self.parameters[k]) for k in sorted(self.parameters.keys()))
        return hash(('Emission', self.primitive.identifier, pkey))


