class Extent:
    """

    """
    def __init__(self, *args):
        self._extents = list()

        if len(args) % 2 == 1:
            ValueError("An uneven number of ranges were passed to the constructor.")
        for i in range(0, len(args), 2):
            self._extents.append((args[i], args[i+1]))



    @property
    def x(self):
        """
        Gives the extent in x-direction.
        """
        return self._extents[0]

    @property
    def y(self):
        return self._extents[1]

    @property
    def z(self):
        return self._extents[2]
