import re


def clean_pipeline_string(individual):
    """ Creates a `pretty` version of the individual string, removing hyperparameter prefixes and the 'data' argument.

    :param individual: Individual of which to return a pretty string representation.
    :return: A string that represents the individual.
    """
    ugly_string = str(individual)
    # Data terminal is found either as '...(data, ....' or '...(data)'
    terminal_signature = '\(data[,)]'
    match = re.search(terminal_signature, ugly_string)
    if match:
        if ugly_string[match.start():match.end()] == '(data)':
            pretty_string = ugly_string[:match.start() + 1] + ugly_string[match.end() - 1:]
        else:
            pretty_string = ugly_string[:match.start() + 1] + ugly_string[match.end():]
        # Remove hyperparameter prefixes: scan to the last period that is before any '=' sign
        pretty_string = ','.join([re.sub(' [^=]+\.', '', sub_string, 1) for sub_string in pretty_string.split(',')])
        # Because some hyperparameters have a prefix and some don't (shared ones), we can't know where spaces are.
        # Remove all spaces and re-insert them only where wanted.
        pretty_string = pretty_string.replace(' ', '')
        pretty_string = pretty_string.replace(',', ', ')
        return pretty_string
    else:
        raise ValueError("All pipeline strings should contain the data terminal.")
