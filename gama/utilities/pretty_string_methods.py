import re


def clean_pipeline_string(individual):
    """ Creates a `pretty` version of the individual string, removing hyperparameter prefixes and the 'data' argument.

    :param individual: Individual of which to return a pretty string representation.
    :return: A string that represents the individual.
    """
    ugly_string = str(individual)
    # Remove the 'data' terminal
    terminal_signature = 'data,'
    if terminal_signature in ugly_string:
        terminal_idx = ugly_string.index(terminal_signature)
        pretty_string = ugly_string[:terminal_idx] + ugly_string[terminal_idx + len(terminal_signature):]
        # Remove hyperparameter prefixes
        pretty_string = re.sub('[ .+\.]', '', pretty_string)
        # Because some hyperparameters have a prefix and some don't (shared ones), we can't know where spaces are.
        # Remove all spaces and re-insert them only where wanted.
        pretty_string = pretty_string.replace(' ', '')
        pretty_string = pretty_string.replace(',', ', ')
        return pretty_string
    else:
        raise ValueError("All pipeline strings should contain the data terminal.")
