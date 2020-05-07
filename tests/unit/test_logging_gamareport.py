from gama.logging.GamaReport import GamaReport


def test_gamareport_from_log():
    """ GamaReport can be constructed from a log that recorded RandomSearch. """
    # We refer to a static log, this makes it independent of other unit tests,
    # but it also makes it independent of the actual changes in gama logging.
    # Cons:
    #   - when logging changes and the static file is not updated, we won't catch it.
    # Pros:
    #   + unit test independence
    #   + backwards compatibility test for GamaReport
    # Perhaps we can/should link to the log file used in the documentation.
    log_file = "tests/data/random_search.log"
    report = GamaReport(logfile=log_file, name=None)
    assert report.name == log_file
    assert "RandomSearch()" == report.search_method
    assert 3 == len(report.phases)
    assert ["preprocessing", "search", "postprocess"] == list(
        map(lambda t: t[0], report.phases)
    )
    assert ["default", "RandomSearch", "NoPostProcessing"] == list(
        map(lambda t: t[1], report.phases)
    )
    assert report.method_data is None, "Random Search has no method data associated."


def test_gamareport_asha_from_log():
    """ GamaReport can be constructed from a log that recorded ASHA. """
    log_file = "tests/data/asha.log"
    report = GamaReport(logfile=log_file, name=None)
    assert report.name == log_file
    assert "AsynchronousSuccessiveHalving()" == report.search_method
    assert 3 == len(report.phases)
    assert ["preprocessing", "search", "postprocess"] == list(
        map(lambda t: t[0], report.phases)
    )
    assert ["default", "AsynchronousSuccessiveHalving", "NoPostProcessing"] == list(
        map(lambda t: t[1], report.phases)
    )
    assert report.method_data is not None, "ASHA has method data associated."


def test_gamareport_asyncEA_from_log():
    """ GamaReport can be constructed from a log that recorded AsyncEA. """
    log_file = "tests/data/async_ea.log"
    report = GamaReport(logfile=log_file, name=None)
    assert report.name == log_file
    assert "AsyncEA()" == report.search_method
    assert 3 == len(report.phases)
    assert ["preprocessing", "search", "postprocess"] == list(
        map(lambda t: t[0], report.phases)
    )
    assert ["default", "AsyncEA", "NoPostProcessing"] == list(
        map(lambda t: t[1], report.phases)
    )
    assert report.method_data is None, "AsyncEA has no method data associated."
