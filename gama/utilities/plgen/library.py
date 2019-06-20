from d3m import index
from gama.genetic_programming.components import DATA_TERMINAL


class Library(object):

    METADATA_FIELDS = dict(
        family='primitive_family'
    )

    def prime(self, substring):
        pass

    def primitives_with_name(self, name):
        matching_names = [n for n in self.all_primitives_names() if name in n]
        return [self.get_primitive_by_name(n) for n in matching_names]

    def primitives_of_type(self, type_, candidates):
        return [p for p in self.all_primitives() if self.primitive_has_type(p, type_, candidates)]

    def primitive_has_type(self, primitive, type_, candidates):
        try:
            mdfield = self.METADATA_FIELDS[type_]
            return self.primitive_metadata(primitive)[mdfield] in candidates
        except KeyError:
            raise NotImplementedError("Cannot search on type %s" % type_)

    def all_primitives_names(self):
        raise NotImplementedError()

    def get_primitive_by_name(self, name):
        raise NotImplementedError()

    def primitive_name(self, primitive):
        raise NotImplementedError()

    def primitive_path(self, primitive):
        raise NotImplementedError()

    def all_primitives(self):
        raise NotImplementedError()

    def primitive_metadata(self, primitive):
        raise NotImplementedError()


class D3MLibrary(Library):

    def prime(self, substring):
        for p in index.search():
            if substring in p:
                index.get_primitive(p)

    def all_primitives_names(self):
        for name in index.search():
            yield name
        for primitive in index.get_loaded_primitives():
            yield primitive.__name__

    def get_primitive_by_name(self, name):
        return index.get_primitive(name)

    def primitive_name(self, primitive):
        return primitive.__name__

    def primitive_path(self, primitive):
        return primitive.metadata.query()['python_path']

    def all_primitives(self):
        return index.get_loaded_primitives()

    def primitive_metadata(self, primitive):
        return primitive.metadata.query()


class GamaPsetLibrary(Library):

    def __init__(self, gama_pset):
        self.gama_pset = gama_pset

    def all_primitives_names(self):
        for sub_pset in self.gama_pset.values():
            for p in sub_pset:
                yield self.primitive_metadata(p)['python_path']
                yield p.__name__

    def get_primitive_by_name(self, name):
        for sub_pset in self.gama_pset.values():
            for p in sub_pset:
                if name in self.primitive_metadata(p)['python_path'] or name in p.__name__:
                    return p
        return None

    def _d3m_primitive(self, primitive):
        return primitive[2].pclass

    def primitive_name(self, primitive):
        return self._d3m_primitive(primitive).__name__

    def primitive_path(self, primitive):
        return self.primitive_metadata(primitive)['python_path']

    def all_primitives(self):
        return self.gama_pset[DATA_TERMINAL] + self.gama_pset['prediction']

    def primitive_metadata(self, primitive):
        return self._d3m_primitive(primitive).metadata.query()



