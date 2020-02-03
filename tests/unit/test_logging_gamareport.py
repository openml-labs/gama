from gama.logging.GamaReport import GamaReport


def test_gamareport_from_log():
    """ Test that a GamaReport can be constructed from a log. """
    # We refer to a static log, this makes it independent of other unit tests, but it also makes it
    # independent of the actual changes in gama logging. This is bad in the sense that if logging changes
    # and the static file is not updated, we won't catch it. This is good in the sense that it achieves unit test
    # independence and backwards incompatability of changes to GamaReport are immediately caught if tested on the
    # old log first.
    # Perhaps we can/should link to the log file used in the documentation.
    log_file = 'tests/data/random_search.log'
    report = GamaReport(logfile=log_file, name=None)
    assert report.name == log_file
    assert 'RandomSearch' == report.search_method
    assert 3 == len(report.phases)
    assert (['preprocessing', 'search', 'postprocess']
            == list(map(lambda t: t[0], report.phases)))
    assert (['default', 'RandomSearch', 'NoPostProcessing']
            == list(map(lambda t: t[1], report.phases)))
    assert report.method_data is None, "Random Search has no method data associated."


def test_gamareport_asha_from_log():
    """ Test that a GamaReport can be constructed from a log and retrieve ASHA specific information. """
    log_file = 'tests/data/asha.log'
    report = GamaReport(logfile=log_file, name=None)
    assert report.name == log_file
    assert 'AsynchronousSuccessiveHalving' == report.search_method
    assert 3 == len(report.phases)
    assert (['preprocessing', 'search', 'postprocess']
            == list(map(lambda t: t[0], report.phases)))
    assert (['default', 'AsynchronousSuccessiveHalving', 'NoPostProcessing']
            == list(map(lambda t: t[1], report.phases)))
    assert report.method_data is not None, "ASHA has method data associated."
