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

    # def batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True,
    #               to_csv: bool = False, output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
    #     """
    #     Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in a DataFrame.
    #     The input must have only 3 columns: date_time, price, & volume.

    #     :param file_path_or_df: (str, iterable of str, or pd.DataFrame) The input data source.
    #     :param verbose: (bool) Flag whether to print messages on each processed batch.
    #     :param to_csv: (bool) Flag for writing results to a CSV file or returning an in-memory DataFrame.
    #     :param output_path: (str) Path to results file, if to_csv = True.

    #     :return: (pd.DataFrame or None) Financial data structure
    #     """
    #     if to_csv is True:
    #         header = True
    #         open(output_path, 'w').close()  # clean output csv file

    #     if verbose:  # pragma: no cover
    #         print('Reading data in batches:')

    #     # Determine the output columns:
    #     # Base columns + additional feature names
    #     cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume',
    #             'cum_ticks', 'cum_dollar_value']
    #     feature_cols = [feature.name for feature in self.additional_features]
    #     cols.extend(feature_cols)

    #     # Read csv in batches
    #     count = 0
    #     final_bars = []
    #     for batch in self._batch_iterator(file_path_or_df):
    #         if verbose:  # pragma: no cover
    #             print('Batch number:', count)

    #         list_bars = self.run(data=batch)

    #         if to_csv is True:
    #             pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
    #             header = False
    #         else:
    #             # Append to bars list
    #             final_bars += list_bars
    #         count += 1

    #     if verbose:  # pragma: no cover
    #         print('Returning bars \n')

    #     # Return a DataFrame
    #     if final_bars:
    #         bars_df = pd.DataFrame(final_bars, columns=cols)
    #         return bars_df

    #     return None

    # def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
    #     """
    #     Generator that yields batches of data frames.
    #     """
    #     if isinstance(file_path_or_df, (list, tuple)):
    #         # Assert format of all files
    #         for file_path in file_path_or_df:
    #             self._read_first_row(file_path)
    #         for file_path in file_path_or_df:
    #             for batch in pd.read_csv(file_path, chunksize=self.batch_size, parse_dates=[0]):
    #                 yield batch

    #     elif isinstance(file_path_or_df, str):
    #         self._read_first_row(file_path_or_df)
    #         for batch in pd.read_csv(file_path_or_df, chunksize=self.batch_size, parse_dates=[0]):
    #             yield batch

    #     elif isinstance(file_path_or_df, pd.DataFrame):
    #         for batch in _crop_data_frame_in_batches(file_path_or_df, self.batch_size):
    #             yield batch

    #     else:
    #         raise ValueError('file_path_or_df is neither string(path to a csv file), '
    #                          'iterable of strings, nor pd.DataFrame')

    def batch_run(self, file_path_or_df: Union[
            str,
            Iterable[Union[str, pd.DataFrame]],
            pd.DataFrame],
            verbose: bool = True, to_csv: bool = False,
            output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Reads CSV file(s), DataFrame(s), or an iterator of DataFrames/file paths in batches and then constructs the financial data structure.
        The input must have only 3 columns: date_time, price, & volume.

        :param file_path_or_df: (str, iterable of str or DataFrame, or pd.DataFrame) The input data source.
        :param verbose: (bool) Flag whether to print messages on each processed batch.
        :param to_csv: (bool) Flag for writing results to a CSV file or returning an in-memory DataFrame.
        :param output_path: (str) Path to results file, if to_csv = True.
        :return: (pd.DataFrame or None) Financial data structure
        """
        if to_csv:
            header = True
            open(output_path, 'w').close()  # Clear output CSV file

        if verbose:  # pragma: no cover
            print('Reading data in batches:')

        # Determine the output columns:
        # Base columns + additional feature names
        cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume',
                'cum_buy_volume', 'cum_ticks', 'cum_dollar_value']
        feature_cols = [feature.name for feature in self.additional_features]
        cols.extend(feature_cols)

        count = 0
        final_bars = []
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:  # pragma: no cover
                print('Processing batch number:', count)

            list_bars = self.run(data=batch)

            if to_csv:
                pd.DataFrame(list_bars, columns=cols).to_csv(
                    output_path, header=header, index=False, mode='a'
                )
                header = False
            else:
                final_bars += list_bars
            count += 1

        if verbose:  # pragma: no cover
            print('Returning bars\n')

        if final_bars:
            return pd.DataFrame(final_bars, columns=cols)
        return None

    def _batch_iterator(self, file_path_or_df: Union[
            str,
            Iterable[Union[str, pd.DataFrame]],
            pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
        """
        Generator that yields batches of DataFrames.
        Supports:
         - A single CSV file path (str)
         - An iterable (e.g., list, tuple, or generator) of CSV file paths or DataFrames
         - A single DataFrame
        """
        # Handle list or tuple explicitly
        if isinstance(file_path_or_df, (list, tuple)):
            if all(isinstance(x, pd.DataFrame) for x in file_path_or_df):
                for df in file_path_or_df:
                    for batch in _crop_data_frame_in_batches(df, self.batch_size):
                        yield batch
            elif all(isinstance(x, str) for x in file_path_or_df):
                for file_path in file_path_or_df:
                    self._read_first_row(file_path)
                for file_path in file_path_or_df:
                    for batch in pd.read_csv(file_path, chunksize=self.batch_size, parse_dates=[0]):
                        yield batch
            else:
                raise ValueError("List or tuple items must all be either DataFrames or strings representing file paths.")

        # Handle a single CSV file path
        elif isinstance(file_path_or_df, str):
            self._read_first_row(file_path_or_df)
            for batch in pd.read_csv(file_path_or_df, chunksize=self.batch_size, parse_dates=[0]):
                yield batch

        # Handle a single DataFrame
        elif isinstance(file_path_or_df, pd.DataFrame):
            for batch in _crop_data_frame_in_batches(file_path_or_df, self.batch_size):
                yield batch

        # Handle any other iterable (e.g., a generator)
        elif isinstance(file_path_or_df, Iterable):
            for element in file_path_or_df:
                if isinstance(element, pd.DataFrame):
                    for batch in _crop_data_frame_in_batches(element, self.batch_size):
                        yield batch
                elif isinstance(element, str):
                    self._read_first_row(element)
                    for batch in pd.read_csv(element, chunksize=self.batch_size, parse_dates=[0]):
                        yield batch
                else:
                    raise ValueError("Elements in the iterable must be either pd.DataFrame or str.")
        else:
            raise ValueError(
                'file_path_or_df must be either a string (path to a CSV file), an iterable of strings/DataFrames, or a pd.DataFrame.'
            )

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
        
    def _get_price_diff(self, price):
        """
        Compute price difference.
        """
        return price - self.prev_price if self.prev_price is not None else 0

    def _get_log_ret(self, price):
        """
        Compute log return.
        """
        return np.log(price / self.prev_price) if self.prev_price is not None else 0

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
    def _update_ticks_in_bar(self, row):
        """
        Update the ticks in the current bar, computing additional columns for tick_df.

        :param row: (array-like) A single tick with [date_time, price, volume].
        """
        date_time, price, volume = row
        price_diff = self._get_price_diff(price)
        log_ret = self._get_log_ret(price)
        tick_sign = self._apply_tick_rule(price)

        # Add the tick data along with computed columns to the list
        self.ticks_in_current_bar.append({
            'date_time': date_time,
            'price': price,
            'volume': volume,
            'price_diff': price_diff,
            'log_ret': log_ret,
            'tick_sign': tick_sign
        })

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


class BaseImbalanceBars(BaseBars):
    """
    Base class for Imbalance Bars (EMA and Const) which implements imbalance bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int,
                 expected_imbalance_window: int, exp_num_ticks_init: int,
                 analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init
        :param analyse_thresholds: (bool) Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a
                                          form of Pandas DataFrame
        """
        BaseBars.__init__(self, metric, batch_size)

        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {'cum_theta': 0, 'expected_imbalance': np.nan, 'exp_num_ticks': exp_num_ticks_init}

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {'num_ticks_bar': [], 'imbalance_array': []}

        if analyse_thresholds is True:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.thresholds['cum_theta'] = 0

    def _extract_bars(self, data: Tuple[dict, pd.DataFrame]) -> list:
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """

        # Iterate over rows
        list_bars = []
        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Bar statistics calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Imbalance calculations
            imbalance = self._get_imbalance(price, signed_tick, volume)
            self.imbalance_tick_statistics['imbalance_array'].append(imbalance)
            self.thresholds['cum_theta'] += imbalance

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and np.isnan(self.thresholds['expected_imbalance']):
                self.thresholds['expected_imbalance'] = self._get_expected_imbalance(
                    self.expected_imbalance_window)

            if self.bars_thresholds is not None:
                self.thresholds['timestamp'] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            # Check expression for possible bar generation
            if (np.abs(self.thresholds['cum_theta']) > self.thresholds['exp_num_ticks'] * np.abs(
                    self.thresholds['expected_imbalance']) if ~np.isnan(self.thresholds['expected_imbalance']) else False):
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
                # Expected number of ticks based on formed bars
                self.thresholds['exp_num_ticks'] = self._get_exp_num_ticks()
                # Get expected imbalance
                self.thresholds['expected_imbalance'] = self._get_expected_imbalance(
                    self.expected_imbalance_window)
                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, window: int):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
        :param window: (int) EWMA window for calculation
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(self.imbalance_tick_statistics['imbalance_array']) < self.thresholds['exp_num_ticks']:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(self.imbalance_tick_statistics['imbalance_array']), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(self.imbalance_tick_statistics['imbalance_array'][-ewma_window:], dtype=float),
                window=ewma_window)[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new run bar is formed
        """


# pylint: disable=too-many-instance-attributes
class BaseRunBars(BaseBars):
    """
    Base class for Run Bars (EMA and Const) which implements run bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int, num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int, analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init
        :param analyse_thresholds: (bool) Flag to return thresholds values (thetas, exp_num_ticks, exp_runs) in Pandas DataFrame
        """
        BaseBars.__init__(self, metric, batch_size)

        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {'cum_theta_buy': 0, 'cum_theta_sell': 0, 'exp_imbalance_buy': np.nan,
                           'exp_imbalance_sell': np.nan, 'exp_num_ticks': exp_num_ticks_init,
                           'exp_buy_ticks_proportion': np.nan, 'buy_ticks_num': 0}

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {'num_ticks_bar': [], 'imbalance_array_buy': [], 'imbalance_array_sell': [],
                                          'buy_ticks_proportion': []}

        if analyse_thresholds:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

        self.warm_up_flag = False

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.thresholds['cum_theta_buy'], self.thresholds['cum_theta_sell'], self.thresholds['buy_ticks_num'] = 0, 0, 0

    def _extract_bars(self, data: Tuple[list, np.ndarray]) -> list:
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (list or np.ndarray) Contains 3 columns - date_time, price, and volume.
        :return: (list) of bars built using the current batch.
        """

        # Iterate over rows
        list_bars = []
        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Bar statistics calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Imbalance calculations
            imbalance = self._get_imbalance(price, signed_tick, volume)

            if imbalance > 0:
                self.imbalance_tick_statistics['imbalance_array_buy'].append(imbalance)
                self.thresholds['cum_theta_buy'] += imbalance
                self.thresholds['buy_ticks_num'] += 1
            elif imbalance < 0:
                self.imbalance_tick_statistics['imbalance_array_sell'].append(abs(imbalance))
                self.thresholds['cum_theta_sell'] += abs(imbalance)

            self.warm_up_flag = np.isnan([self.thresholds['exp_imbalance_buy'], self.thresholds[
                'exp_imbalance_sell']]).any()  # Flag indicating that one of imbalances is not counted (warm-up)

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and self.warm_up_flag:
                self.thresholds['exp_imbalance_buy'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_buy'], self.expected_imbalance_window, warm_up=True)
                self.thresholds['exp_imbalance_sell'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_sell'], self.expected_imbalance_window,
                    warm_up=True)

                if bool(np.isnan([self.thresholds['exp_imbalance_buy'],
                                  self.thresholds['exp_imbalance_sell']]).any()) is False:
                    self.thresholds['exp_buy_ticks_proportion'] = self.thresholds['buy_ticks_num'] / \
                                                                  self.cum_statistics[
                                                                      'cum_ticks']

            if self.bars_thresholds is not None:
                self.thresholds['timestamp'] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            # Check expression for possible bar generation
            max_proportion = max(
                self.thresholds['exp_imbalance_buy'] * self.thresholds['exp_buy_ticks_proportion'],
                self.thresholds['exp_imbalance_sell'] * (1 - self.thresholds['exp_buy_ticks_proportion']))

            # Check expression for possible bar generation
            max_theta = max(self.thresholds['cum_theta_buy'], self.thresholds['cum_theta_sell'])
            if max_theta > self.thresholds['exp_num_ticks'] * max_proportion and not np.isnan(max_proportion):
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)

                self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
                self.imbalance_tick_statistics['buy_ticks_proportion'].append(
                    self.thresholds['buy_ticks_num'] / self.cum_statistics['cum_ticks'])

                # Expected number of ticks based on formed bars
                self.thresholds['exp_num_ticks'] = self._get_exp_num_ticks()

                # Expected buy ticks proportion based on formed bars
                exp_buy_ticks_proportion = ewma(
                    np.array(self.imbalance_tick_statistics['buy_ticks_proportion'][-self.num_prev_bars:], dtype=float),
                    self.num_prev_bars)[-1]
                self.thresholds['exp_buy_ticks_proportion'] = exp_buy_ticks_proportion

                # Get expected imbalance
                self.thresholds['exp_imbalance_buy'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_buy'], self.expected_imbalance_window)
                self.thresholds['exp_imbalance_sell'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_sell'], self.expected_imbalance_window)

                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False):
        """
        Advances in Financial Machine Learning, page 29.

        Calculates the expected imbalance: 2P[b_t=1]-1, using a EWMA.

        :param array: (list) of imbalances
        :param window: (int) EWMA window for calculation
        :parawm warm_up: (bool) flag of whether warm up period passed
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(array) < self.thresholds['exp_num_ticks'] and warm_up is True:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(array), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(array[-ewma_window:], dtype=float),
                window=ewma_window)[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new imbalance bar is formed
        """
