import plac
from plgen.manager import Manager

def main(fname, name):
    mgr = Manager()
    mgr.parse_file(fname)
#    mgr.dump_fa(name)
    for _ in range(5):
        seq = mgr.generate(name)
        print(seq)
        print(mgr.matches(name, seq))
        # Mangle it for fun
        seq += [seq[0]]
        # Should not match now
        print(mgr.matches(name, seq))

plac.call(main)

