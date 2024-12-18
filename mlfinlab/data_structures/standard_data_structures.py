"""
Advances in Financial Machine Learning, Marcos Lopez de Prado
Chapter 2: Financial Data Structures

This module contains the functions to help users create structured financial data from raw unstructured data,
in the form of time, tick, volume, and dollar bars.

These bars are used throughout the text book (Advances in Financial Machine Learning, By Marcos Lopez de Prado, 2018,
pg 25) to build the more interesting features for predicting financial time series data.

These financial data structures have better statistical properties when compared to those based on fixed time interval
sampling. A great paper to read more about this is titled: The Volume Clock: Insights into the high frequency paradigm,
Lopez de Prado, et al.

Many of the projects going forward will require Dollar and Volume bars.
"""

# Imports
from typing import Union, Iterable, Optional

import numpy as np
import pandas as pd

from mlfinlab.data_structures.base_bars import BaseBars


class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, metric, batch_size)

        # Threshold at which to sample
        self.threshold = threshold

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """

        # Iterate over rows
        list_bars = []

        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # If threshold reached then take a sample
            if self.cum_statistics[self.metric] >= self.threshold:  # pylint: disable=eval-used
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                # Reset cache
                self._reset_cache()
        return list_bars


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: float = 70000000, batch_size: int = 20000000,
                    verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float) A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of dollar bars
    """

    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: float = 70000000, batch_size: int = 20000000,
                    verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float) A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of volume bars
    """
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: float = 70000000, batch_size: int = 20000000,
                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                             in the format[date_time, price, volume]
    :param threshold: (float) A cumulative value above this threshold triggers a sample to be taken.
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of volume bars
    """
    bars = StandardBars(metric='cum_ticks',
                        threshold=threshold, batch_size=batch_size)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars



class RelativePriceBars(BaseBars):
    """
    This class creates bars based on a relative price threshold (percentage).
    Once the price moves away from the open_price of the current bar
    by at least `threshold` * open_price (in either direction), a new bar is formed.

    Attributes:
        threshold: (float) The relative price threshold. For example:
                   threshold=0.01 means a 1% move from the open_price triggers a new bar.
    """

    def __init__(self, threshold: float, batch_size: int = 20000000):
        """
        Constructor

        :param threshold: (float) Relative price movement threshold that triggers a bar creation.
                          e.g. 0.01 means 1% move from the open price.
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        super().__init__(metric='rel_price', batch_size=batch_size)
        self.threshold = threshold
        self._reset_cache()

    def _reset_cache(self):
        """
        Resets the cache for the new bar creation.
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {
            'cum_ticks': 0,
            'cum_dollar_value': 0,
            'cum_volume': 0,
            'cum_buy_volume': 0
        }

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        Iterate over rows and construct relative price threshold bars.

        :param data: (tuple or np.ndarray) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """
        list_bars = []

        for row in data:
            date_time = row[0]
            self.tick_num += 1
            price = float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            # If this is the first tick of a new bar, set the open price.
            if self.open_price is None:
                self.open_price = price

            # Update high/low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Update cumulative stats
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Compute the thresholds
            upper_bound = self.open_price * (1 + self.threshold)
            lower_bound = self.open_price * (1 - self.threshold)

            # Check if the relative price threshold has been breached
            if price >= upper_bound or price <= lower_bound:
                # Create a new bar
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)
                # Reset the cache for the next bar
                self._reset_cache()

        return list_bars


def get_relative_price_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame],
                            threshold: float,
                            batch_size: int = 20000000,
                            verbose: bool = True,
                            to_csv: bool = False,
                            output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Creates bars based on relative price moves: 
    date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    For example, if threshold=0.01, this means that once the price moves 1% above or below
    the bar's open price, a new bar is formed.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path(s) to the csv file(s) or DataFrame 
                            containing raw tick data in the format [date_time, price, volume]
    :param threshold: (float) The relative price movement threshold. e.g. 0.01 = 1%
    :param batch_size: (int) Number of rows per batch.
    :param verbose: (bool) Print out batch information.
    :param to_csv: (bool) Save bars to csv after every batch run.
    :param output_path: (str) Path to csv file, if to_csv is True.
    :return: (pd.DataFrame) Dataframe of relative price threshold bars.
    """
    bars = RelativePriceBars(threshold=threshold, batch_size=batch_size)
    relative_price_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return relative_price_bars
