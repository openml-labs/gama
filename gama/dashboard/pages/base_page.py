import abc


class BasePage(abc.ABC):

    def __init__(self, name: str, alignment: int):
        """ Defines the basic behavior of a page.

        Parameters
        ----------
        name: str
            Name of the page, displayed in the tab.

        alignment: int
            Defines the order of tabs.
            Positive numbers are aligned to the left, negative to the right.
            Within the groups, bigger numbers are placed to the right.
            E.g.: [0][1][2] ... [-2][-1]
        """
        self.name = name
        self.alignment = alignment

    @abc.abstractmethod
    def build_page(self):
        raise NotImplementedError
