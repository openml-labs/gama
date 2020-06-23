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
    log_dir = "tests/data/RandomSearch"
    report = GamaReport(log_dir)
    assert report.name == "RandomSearch"
    assert "RandomSearch" == report.search_method
    assert 3 == len(report.phases)
    assert ["preprocessing", "search", "postprocess"] == list(
        map(lambda t: t[0], report.phases)
    )
    assert ["default", "RandomSearch", "NoPostProcessing"] == list(
        map(lambda t: t[1], report.phases)
    )


def test_gamareport_asha_from_log():
    """ GamaReport can be constructed from a log that recorded ASHA. """
    log_dir = "tests/data/ASHA"
    report = GamaReport(log_dir)
    assert report.name == "ASHA"
    assert "AsynchronousSuccessiveHalving" == report.search_method
    assert 3 == len(report.phases)
    assert ["preprocessing", "search", "postprocess"] == list(
        map(lambda t: t[0], report.phases)
    )
    assert ["default", "AsynchronousSuccessiveHalving", "NoPostProcessing"] == list(
        map(lambda t: t[1], report.phases)
    )


def test_gamareport_asyncEA_from_log():
    """ GamaReport can be constructed from a log that recorded AsyncEA. """
    log_dir = "tests/data/AsyncEA"
    report = GamaReport(log_dir)
    assert report.name == "AsyncEA"
    assert "AsyncEA" == report.search_method
    assert 3 == len(report.phases)
    assert ["preprocessing", "search", "postprocess"] == list(
        map(lambda t: t[0], report.phases)
    )
    assert ["default", "AsyncEA", "NoPostProcessing"] == list(
        map(lambda t: t[1], report.phases)
    )
