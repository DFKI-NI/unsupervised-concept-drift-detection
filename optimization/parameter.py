from typing import Any, List, Optional


class Parameter:
    def __init__(
        self,
        name: str,
        *,
        value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        n_values: Optional[Any] = None,
        step_size: Optional[Any] = None,
        values: Optional[List[Any]] = None,
    ):
        """
        Init a parameter. Not all combinations are possible, check the different generators for supported options.

        :param name: the parameter's name
        :param value: the value if the parameter can assume a single value
        :param max_value: the max value if multiple values are possible
        :param n_values: the number of steps between value and max value
        :param step_size: the step size between values
        :param values: the values as a list
        """
        self.name = name
        self.value = value
        self.max_value = max_value
        self.n_values = n_values
        self.step_size = step_size
        self.values = values

    def _select_generator(self):
        """
        Select the generator used to yield configuration parameter values based on the parameters provided at
        initialization.

        :return: the generator
        """
        if self.n_values is not None and self.step_size is None:
            return self.n_values_generator()
        elif self.step_size is not None and self.n_values is None:
            return self.step_size_generator()
        elif self.step_size is not None and self.n_values is not None:
            return self.n_values_step_size_generator()
        elif self.values is not None:
            return self.list_generator()
        else:
            return self.value_generator()

    def list_generator(self):
        """
        The parameter is configured to yield from a list of values. Values are provided at initialization.

        :return: the values
        """
        for item in self.values:
            yield item

    def value_generator(self):
        """
        Yields value and max value if max value is not None. Set only value and optionally max value at initialization.

        :return: the value
        """
        yield self.value
        if self.max_value is not None:
            yield self.max_value

    def n_values_generator(self):
        """
        Yields n equidistant values using value and max value as minimum and maximum respectively. Provide value,
        max_value and n_values at initialization.

        :return: the values
        """
        step_size = (self.max_value - self.value) / (self.n_values - 1)
        for i in range(self.n_values):
            yield self.value + i * step_size

    def step_size_generator(self):
        """
        Yields values with a distance of step_size until max_value is exceeded starting with value. Provide value,
        max_value and step_size at initialization.

        :return: the values
        """
        i = 0
        while self.value + i * self.step_size <= self.max_value:
            yield self.value + i * self.step_size
            i += 1

    def n_values_step_size_generator(self):
        """
        Yields n_values with the given step_size starting at value. Provide value, n_values and step_size at
        initialization.

        :return: the values
        """
        for i in range(self.n_values):
            yield self.value + i * self.step_size

    def __iter__(self):
        return self._select_generator()
