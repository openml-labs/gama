import re
import os.path

from .tokentest import TokenTestExpression
from .regex import RegexExpression
from .library import D3MLibrary

class Manager(object):

    def __init__(self, **args):
        self.fas = {}
        self.fa_expressions = {}
        self.tests = {}
        self.test_expressions = {}
        self.imports = {}
        self.import_expressions = {}
        if 'library' in args:
            self.library = args['library']
        else:
            self.library = D3MLibrary()
    
    def parse_statement(self, stmt):
        "Parse a single statement"
        # -> expression
        # : TokenTest
        m = re.match('(\w+)\s*(->|:)\s*(.*)', stmt)
        if not m:
            raise ValueError("Badly formed statement: %s" % stmt)

        name = m.group(1)
        op = m.group(2)
        expr = m.group(3).strip()

        if op == ':':
            self.tests[name] = TokenTestExpression(manager=self).parse(expr)
            self.test_expressions[name] = expr
        elif op == '->':
            #print "Parsing", stmt
            fa = RegexExpression(string=expr, manager=self).parse().fa()
            fa.name = name
            fa.parent = self
            self.fas[name] = fa
            self.fa_expressions[name] = expr
        return name

    def parse_block(self, block):
        """Parse string with patterns, represent as internal set of FAs."""
        lines = re.split('\n', block)
        lines = [ re.sub('#.*', '', line) for line in lines ]
        lines = [ re.sub('\s*$', '', line) for line in lines ]
        block = '\n'.join(lines)

        def statements(block):
            statement = None
            for line in re.split('\n', block):
                if not re.search('\S', line):   # Empty line
                    if statement is not None:
                        statement = re.sub('\s+', ' ', statement)
                        yield statement
                    statement = None
                elif re.match('\s', line):      # Indented line
                    statement += line
                else:                           # Left-flush
                    if statement is not None:
                        statement = re.sub('\s+', ' ', statement)
                        yield statement
                    statement = line
            if statement is not None:
                statement = re.sub('\s+', ' ', statement)
                yield statement
        
        for stmt in statements(block):
            self.parse_statement(stmt)

    def file_contents(self, fname):
        with open(fname, "r") as fh:
            contents = fh.read()
        return contents

    def parse_file(self, fname):
        """Parse file with patterns, represent as internal set of FAs."""
        self.source_file = fname
        block = self.file_contents(fname)
        self.parse_block(block)

    def import_file(self, fname):
        path = self.resolve_path(fname)
        subm = VRManager()
        subm.parse_file(path)
        return subm

    def resolve_path(self, fname):
        # First things first.  Resolve fname if not absolute.
        if os.path.abspath(fname) == fname:  # Absolute path
            path = fname
        elif os.path.exists(fname):  # Resolvable from cwd
            path = os.path.abspath(fname)
        else:  # Attempt resolve script dir
            mydir = os.path.dirname(self.source_file)
            path = os.path.join(mydir, fname)
            if not os.path.exists(path):
                raise GenericException(msg="Can't resolve '%s'" % fname)
        return path

    def dump(self):
        """Dump all FAs to stdout."""
        for id, fa in self.fas.items():
            print("FA", id)
            fa.dump()

    def dump_fa(self, which):
        """Dump the FA with the given id to stdout."""
        fa = self.fas[which]
        fa.dump()

    def get_test(self, which):
        m = re.match('(\w+)\.(.*)', which)
        if m:                               # Imported expression
            import_name = m.group(1)
            mgr = self.imports[import_name]
            return mgr.get_test(m.group(2))
        else:
            return self.tests[which]

    def get_fa(self, which):
        try:
            return self.fas[which]
        except KeyError:
            return None

    def get_test_expressions(self):
        return sorted(self.test_expressions.items(), key=lambda x: x[0])

    def get_fa_expressions(self):
        return sorted(self.fa_expressions.items(), key=lambda x: x[0])

    def defined_extractors(self):
        return set(e for e in tab.keys() for tab in 
                   (self.test_expressions, 
                    self.fa_expressions))

    def generate(self, name):
        fa = self.get_fa(name)
        if fa is not None:
            return fa.generate()
        test = self.get_test(name)
        return test.generate()
    
    def matches(self, name, seq):
        fa = self.get_fa(name)
        return fa.matches(seq)



