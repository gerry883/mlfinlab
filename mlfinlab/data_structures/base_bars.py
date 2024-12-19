"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional

import numpy as np
import pandas as pd

from mlfinlab.util.fast_ewma import ewma


def _crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> list:
    # Splits df into chunks of chunksize
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object


class BaseBars(ABC):
    """
    Abstract base class which contains the structure shared between various standard and information-driven bars.
    """

    def __init__(self, metric: str, batch_size: int = 2e7, additional_features=None):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: 'dollar_imbalance'.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param additional_features: (list) A list of objects each containing:
                                    - name: str, the name of the feature
                                    - compute(df): a method that computes the feature from a DataFrame of ticks
        """
        # Base properties
        self.metric = metric
        self.batch_size = batch_size
        self.prev_tick_rule = 0

        # Cache properties
        self.open_price, self.prev_price, self.close_price = None, None, None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.tick_num = 0  # Tick number when bar was formed

        # Batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache

        # Additional features
        if additional_features is None:
            additional_features = []
        self.additional_features = additional_features
        self.ticks_in_current_bar = []
        self.computed_additional_features = []

    def batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True,
                  to_csv: bool = False, output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in a DataFrame.
        The input must have only 3 columns: date_time, price, & volume.

        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) The input data source.
        :param verbose: (bool) Flag whether to print messages on each processed batch.
        :param to_csv: (bool) Flag for writing results to a CSV file or returning an in-memory DataFrame.
        :param output_path: (str) Path to results file, if to_csv = True.

        :return: (pd.DataFrame or None) Financial data structure
        """
        if to_csv is True:
            header = True
            open(output_path, 'w').close()  # clean output csv file

        if verbose:  # pragma: no cover
            print('Reading data in batches:')

        # Determine the output columns:
        # Base columns + additional feature names
        cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume',
                'cum_ticks', 'cum_dollar_value']
        feature_cols = [feature.name for feature in self.additional_features]
        cols.extend(feature_cols)

        # Read csv in batches
        count = 0
        final_bars = []
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:  # pragma: no cover
                print('Batch number:', count)

            list_bars = self.run(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                # Append to bars list
                final_bars += list_bars
            count += 1

        if verbose:  # pragma: no cover
            print('Returning bars \n')

        # Return a DataFrame
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        return None

    def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
        """
        Generator that yields batches of data frames.
        """
        if isinstance(file_path_or_df, (list, tuple)):
            # Assert format of all files
            for file_path in file_path_or_df:
                self._read_first_row(file_path)
            for file_path in file_path_or_df:
                for batch in pd.read_csv(file_path, chunksize=self.batch_size, parse_dates=[0]):
                    yield batch

        elif isinstance(file_path_or_df, str):
            self._read_first_row(file_path_or_df)
            for batch in pd.read_csv(file_path_or_df, chunksize=self.batch_size, parse_dates=[0]):
                yield batch

        elif isinstance(file_path_or_df, pd.DataFrame):
            for batch in _crop_data_frame_in_batches(file_path_or_df, self.batch_size):
                yield batch

        else:
            raise ValueError('file_path_or_df is neither string(path to a csv file), '
                             'iterable of strings, nor pd.DataFrame')

    def _read_first_row(self, file_path: str):
        """
        Reads first row to assert format.
        """
        first_row = pd.read_csv(file_path, nrows=1)
        self._assert_csv(first_row)

    def run(self, data: Union[list, tuple, pd.DataFrame]) -> list:
        """
        Processes a given data set (list, tuple, or DataFrame) of ticks
        and returns a list of bars.
        """
        if isinstance(data, (list, tuple)):
            values = data
        elif isinstance(data, pd.DataFrame):
            values = data.values
        else:
            raise ValueError('data is neither list nor tuple nor pd.DataFrame')

        list_bars = self._extract_bars(data=values)
        self.flag = True
        return list_bars

    @abstractmethod
    def _extract_bars(self, data: np.ndarray) -> list:
        """
        Must be implemented in child classes.
        This method creates bars using the data provided.
        - For each tick:
            - Call self._update_ticks_in_bar(tick_row)
            - When a bar is formed:
                self._compute_additional_features()
                self._create_bars(...)
                self._reset_ticks_in_bar()
                self._reset_computed_additional_features()
        """
        pass

    @abstractmethod
    def _reset_cache(self):
        """
        Must be implemented in child classes.
        Describes how the cache should be reset when a new bar is sampled.
        """
        pass

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Tests if the CSV format is correct: date_time, price, volume.
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            raise ValueError('csv file, column 0, not a date time format:',
                             test_batch.iloc[0, 0])

    def _update_high_low(self, price: float) -> Tuple[float, float]:
        """
        Update the high and low prices based on the current price.
        """
        high_price = price if price > self.high_price else self.high_price
        low_price = price if price < self.low_price else self.low_price
        return high_price, low_price

    def _create_bars(self, date_time: str, price: float, high_price: float, low_price: float, list_bars: list) -> None:
        """
        Construct a bar with the given data and append it to list_bars.
        Also append computed additional features.
        """
        open_price = self.open_price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price

        volume = self.cum_statistics['cum_volume']
        cum_buy_volume = self.cum_statistics['cum_buy_volume']
        cum_ticks = self.cum_statistics['cum_ticks']
        cum_dollar_value = self.cum_statistics['cum_dollar_value']

        # Append bar data + additional features
        row = [date_time, self.tick_num, open_price, high_price, low_price, close_price,
               volume, cum_buy_volume, cum_ticks, cum_dollar_value] + self.computed_additional_features
        list_bars.append(row)

    def _apply_tick_rule(self, price: float) -> int:
        """
        Apply the tick rule to determine the sign of the tick.
        """
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        self.prev_price = price
        return signed_tick

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Compute the imbalance based on the chosen metric.
        """
        if self.metric in ['tick_imbalance', 'tick_run']:
            imbalance = signed_tick
        elif self.metric in ['dollar_imbalance', 'dollar_run']:
            imbalance = signed_tick * volume * price
        elif self.metric in ['volume_imbalance', 'volume_run']:
            imbalance = signed_tick * volume
        else:
            raise ValueError('Unknown imbalance metric, possible values are tick/dollar/volume imbalance/run')
        return imbalance

    # ---------------------------
    # Additional Features Methods
    # ---------------------------
    def _update_ticks_in_bar(self, row: np.ndarray) -> None:
        """
        Maintain the list of ticks forming the current bar
        :param row: (ndarray) A single tick with [date_time, price, volume]
        """
        if self.additional_features:
            # Convert row to a dict or a Series with named fields for convenience
            # Let's assume the input row is structured as [date_time, price, volume]
            # Convert to dict for easier handling
            tick_data = {
                'date_time': row[0],
                'price': row[1],
                'volume': row[2]
            }
            self.ticks_in_current_bar.append(tick_data)

    def _reset_ticks_in_bar(self):
        """
        Reset the list of ticks for the next bar
        """
        self.ticks_in_current_bar = []

    def _compute_additional_features(self):
        """
        Compute additional features based on the ticks in the current bar.
        """
        computed_additional_features = []

        if self.additional_features and self.ticks_in_current_bar:
            tick_df = pd.DataFrame(self.ticks_in_current_bar)
            for feature in self.additional_features:
                computed_additional_features.append(feature.compute(tick_df))

        self.computed_additional_features = computed_additional_features

    def _reset_computed_additional_features(self):
        self.computed_additional_features = []
