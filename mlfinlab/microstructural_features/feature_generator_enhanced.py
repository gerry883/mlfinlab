"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features.

This module defines a class to generate microstructural features after bars have been formed.

Available Built-in Features for Selection (pass these strings in `selected_features`):
-------------------------------------------------------------------------------------
- "avg_tick_size"    : Average trade size within the bar.
- "tick_rule_sum"    : Sum of signed ticks based on price movements.
- "vwap"             : Volume Weighted Average Price.
- "kyle_lambda"       : Kyle's Lambda - measures price impact of trades.
- "amihud_lambda"     : Amihud's Lambda - illiquidity measure using log returns and volume.
- "hasbrouck_lambda"  : Hasbrouck's Lambda - liquidity measure using signed ticks and log returns.
- "entropy"           : Entropy metrics (Shannon, Plug-in, Lempel-Ziv, Konto) for encoded tick rule.
                       If `volume_encoding` or `pct_encoding` are provided, volume and pct entropies 
                       will also be computed.

Custom (Additional) Features:
-----------------------------
Users can provide a list of custom feature objects to `additional_features`, each having a `compute()` method
that takes a DataFrame of tick data for the current bar and returns a computed value.

Example Usage:
--------------
custom_features = [MyCustomFeatureClass(), AnotherCustomFeatureClass()]
generator = MicrostructuralFeaturesGenerator(
    trades_input="tick_data.csv",
    tick_num_series=pd.Series([100, 200, 300]),
    selected_features=["vwap", "amihud_lambda", "entropy"],
    additional_features=custom_features
)

features_df = generator.get_features(verbose=True)
print(features_df)
"""

import pandas as pd
import numpy as np
from mlfinlab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy, \
    get_konto_entropy
from mlfinlab.microstructural_features.encoding import encode_array
from mlfinlab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap
from mlfinlab.microstructural_features.encoding import encode_tick_rule_array
from mlfinlab.util.misc import crop_data_frame_in_batches


# pylint: disable=too-many-instance-attributes

class MicrostructuralFeaturesGeneratorEnhanced:
    """
    Class which is used to generate inter-bar features when bars are already compressed.

    Available Built-in Features for Selection:
    -----------------------------------------
    - "avg_tick_size"    : Average trade size within the bar.
    - "tick_rule_sum"    : Sum of signed ticks based on price movements.
    - "vwap"             : Volume Weighted Average Price.
    - "kyle_lambda"      : Kyle's Lambda - measures price impact of trades.
    - "amihud_lambda"    : Amihud's Lambda - illiquidity measure using log returns and volume.
    - "hasbrouck_lambda" : Hasbrouck's Lambda - liquidity measure using signed ticks and log returns.
    - "entropy"          : Entropy metrics (Shannon, Plug-in, Lempel-Ziv, Konto) for encoded tick rule.
                           If `volume_encoding` or `pct_encoding` are provided, volume and pct entropies
                           will also be computed.

    Custom Features:
    ----------------
    Additional custom features can be computed by passing a list of feature objects to `additional_features`.
    Each feature object should implement a `compute()` method that takes the current bar's tick data as a DataFrame
    and returns the computed value.

    :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                         in the format [date_time, price, volume].
    :param tick_num_series: (pd.Series) Series of tick numbers where bars were formed.
    :param batch_size: (int) Number of rows to read in from the csv, per batch.
    :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used for entropy calculations.
    :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used for entropy calculations.
    :param selected_features: (list) List of built-in features to compute. 
                              If None, all features are computed.
                              Example: ["vwap", "entropy", "kyle_lambda"]
    :param additional_features: (list) List of custom feature objects with a `compute()` method.
                                Example: [MyCustomFeature()]
    """

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None,
                 selected_features: list = None, additional_features: list = None):

        if isinstance(trades_input, str):
            self.generator_object = pd.read_csv(trades_input, chunksize=batch_size, parse_dates=[0])
            # Read in the first row & assert format
            first_row = pd.read_csv(trades_input, nrows=1)
            self._assert_csv(first_row)
        elif isinstance(trades_input, pd.DataFrame):
            self.generator_object = crop_data_frame_in_batches(trades_input, batch_size)
        else:
            raise ValueError('trades_input is neither string(path to a csv file) nor pd.DataFrame')

        # Base properties
        self.tick_num_generator = iter(tick_num_series)
        self.current_bar_tick_num = self.tick_num_generator.__next__()

        # Cache properties
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []

        # Entropy properties
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding
        self.entropy_types = ['shannon', 'plug_in', 'lempel_ziv', 'konto']

        # Batch_run properties
        self.prev_price = None
        self.prev_tick_rule = 0
        self.tick_num = 0

        # Feature selection and additional features
        self.selected_features = selected_features
        self.additional_features = additional_features
        self.computed_additional_features = []

    def get_features(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file of ticks or DataFrame in batches and then constructs corresponding microstructural features:
        Including average tick size, tick rule sum, VWAP, Kyle lambda, Amihud lambda, Hasbrouck lambda, and entropies 
        (if encodings are provided).

        Available Built-in Features for Selection:
        -----------------------------------------
        - "avg_tick_size"
        - "tick_rule_sum"
        - "vwap"
        - "kyle_lambda"
        - "amihud_lambda"
        - "hasbrouck_lambda"
        - "entropy"

        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (str) Path to results file, if to_csv = True
        :return: (DataFrame or None) Microstructural features for bar index
        """

        if to_csv is True:
            header = True  # if to_csv is True, header should be written on the first batch only
            open(output_path, 'w').close()  # Clean output csv file

        # Prepare column headers based on selected features
        cols = ['date_time']

        # Conditionally add columns based on selected features
        # If selected_features is None, all features are computed
        if self.selected_features is None or 'avg_tick_size' in self.selected_features:
            cols.append('avg_tick_size')

        if self.selected_features is None or 'tick_rule_sum' in self.selected_features:
            cols.append('tick_rule_sum')

        if self.selected_features is None or 'vwap' in self.selected_features:
            cols.append('vwap')

        if self.selected_features is None or 'kyle_lambda' in self.selected_features:
            cols += ['kyle_lambda', 'kyle_lambda_t_value']

        if self.selected_features is None or 'amihud_lambda' in self.selected_features:
            cols += ['amihud_lambda', 'amihud_lambda_t_value']

        if self.selected_features is None or 'hasbrouck_lambda' in self.selected_features:
            cols += ['hasbrouck_lambda', 'hasbrouck_lambda_t_value']

        if self.selected_features is None or 'entropy' in self.selected_features:
            # Tick rule entropy
            for en_type in self.entropy_types:
                cols.append('tick_rule_entropy_' + en_type)

            # Volume entropy
            if self.volume_encoding is not None:
                for en_type in self.entropy_types:
                    cols.append('volume_entropy_' + en_type)

            # Pct entropy
            if self.pct_encoding is not None:
                for en_type in self.entropy_types:
                    cols.append('pct_entropy_' + en_type)

        # Additional features - column names depend on user-defined features
        # Assuming user-defined features return a single value each
        if self.additional_features:
            for i, _ in enumerate(self.additional_features, start=1):
                cols.append(f'additional_feature_{i}')

        final_bars = []
        count = 0

        # Read csv in batches
        for batch in self.generator_object:
            if verbose:  # pragma: no cover
                print('Batch number:', count)

            list_bars, stop_flag = self._extract_bars(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                # Append to bars list
                final_bars += list_bars
            count += 1

            # End of bar index, no need to calculate further
            if stop_flag is True:
                break

        # Return a DataFrame if not writing to CSV
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        return None

    def _reset_cache(self):
        """
        Reset price_diff, trade_size, tick_rule, log_ret arrays to empty when bar is formed and features are
        calculated
        """
        self.price_diff = []
        self.trade_size = []
        self.tick_rule = []
        self.dollar_size = []
        self.log_ret = []
        self._reset_computed_additional_features()

    def _extract_bars(self, data):
        """
        Iterates over the provided data (ticks) and checks if a bar is formed. 
        If a bar is formed, compute features and reset caches.

        :param data: (pd.DataFrame) Containing columns [date_time, price, volume]
        """

        list_bars = []

        for row in data.values:
            # Set variables
            date_time = row[0]
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            self.tick_num += 1

            # Derivative variables
            price_diff = self._get_price_diff(price)
            log_ret = self._get_log_ret(price)

            self.price_diff.append(price_diff)
            self.trade_size.append(volume)
            self.tick_rule.append(signed_tick)
            self.dollar_size.append(dollar_value)
            self.log_ret.append(log_ret)

            self.prev_price = price

            # If reached the current bar tick_num
            if self.tick_num >= self.current_bar_tick_num:
                self._get_bar_features(date_time, list_bars)

                # Take the next tick number
                try:
                    self.current_bar_tick_num = self.tick_num_generator.__next__()
                except StopIteration:
                    return list_bars, True  # Looped through all bar index
                # Reset cache
                self._reset_cache()
        return list_bars, False

    def _get_bar_features(self, date_time: pd.Timestamp, list_bars: list):
        """
        Calculate selected inter-bar features (and additional user-defined features) for the completed bar.

        :param date_time: (pd.Timestamp) When bar was formed.
        :param list_bars: (list) Previously formed bars (results will be appended here).
        """
        features = [date_time]

        # Compute selected built-in features
        # If no selection, compute all by default
        if self.selected_features is None or 'avg_tick_size' in self.selected_features:
            features.append(get_avg_tick_size(self.trade_size))

        if self.selected_features is None or 'tick_rule_sum' in self.selected_features:
            features.append(sum(self.tick_rule))

        if self.selected_features is None or 'vwap' in self.selected_features:
            features.append(vwap(self.dollar_size, self.trade_size))

        if self.selected_features is None or 'kyle_lambda' in self.selected_features:
            features.extend(get_trades_based_kyle_lambda(self.price_diff, self.trade_size, self.tick_rule))

        if self.selected_features is None or 'amihud_lambda' in self.selected_features:
            features.extend(get_trades_based_amihud_lambda(self.log_ret, self.dollar_size))

        if self.selected_features is None or 'hasbrouck_lambda' in self.selected_features:
            features.extend(get_trades_based_hasbrouck_lambda(self.log_ret, self.dollar_size, self.tick_rule))

        # Entropy features
        if self.selected_features is None or 'entropy' in self.selected_features:
            encoded_tick_rule_message = encode_tick_rule_array(self.tick_rule)
            features.append(get_shannon_entropy(encoded_tick_rule_message))
            features.append(get_plug_in_entropy(encoded_tick_rule_message))
            features.append(get_lempel_ziv_entropy(encoded_tick_rule_message))
            features.append(get_konto_entropy(encoded_tick_rule_message))

            if self.volume_encoding is not None:
                message = encode_array(self.trade_size, self.volume_encoding)
                features.append(get_shannon_entropy(message))
                features.append(get_plug_in_entropy(message))
                features.append(get_lempel_ziv_entropy(message))
                features.append(get_konto_entropy(message))

            if self.pct_encoding is not None:
                message = encode_array(self.log_ret, self.pct_encoding)
                features.append(get_shannon_entropy(message))
                features.append(get_plug_in_entropy(message))
                features.append(get_lempel_ziv_entropy(message))
                features.append(get_konto_entropy(message))

        # Compute additional custom features
        custom_features = self._compute_additional_features()
        features.extend(custom_features)

        list_bars.append(features)

    def _compute_additional_features(self) -> list:
        """
        Compute custom user-defined features provided as additional_features.

        :return: (list) Computed values for additional features.
        """
        computed_features = []
        if self.additional_features:
            tick_df = pd.DataFrame({
                'price_diff': self.price_diff,
                'trade_size': self.trade_size,
                'tick_rule': self.tick_rule,
                'log_ret': self.log_ret,
                'dollar_size': self.dollar_size,
            })
            for feature in self.additional_features:
                computed_features.append(feature.compute(tick_df))
        return computed_features

    def _reset_computed_additional_features(self):
        """
        Reset computed additional features after each bar is processed.
        """
        self.computed_additional_features = []

    def _apply_tick_rule(self, price: float) -> int:
        """
        Advances in Financial Machine Learning, page 29.

        Applies the tick rule

        :param price: (float) Price at time t
        :return: (int) The signed tick
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

        return signed_tick

    def _get_price_diff(self, price: float) -> float:
        """
        Get price difference between ticks

        :param price: (float) Price at time t
        :return: (float) Price difference
        """
        if self.prev_price is not None:
            price_diff = price - self.prev_price
        else:
            price_diff = 0  # First diff is assumed 0
        return price_diff

    def _get_log_ret(self, price: float) -> float:
        """
        Get log return between ticks

        :param price: (float) Price at time t
        :return: (float) Log return
        """
        if self.prev_price is not None:
            log_ret = np.log(price / self.prev_price)
        else:
            log_ret = 0  # First return is assumed 0
        return log_ret

    @staticmethod
    def _assert_csv(test_batch):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) the first row of the dataset.
        :return: (None)
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            print('csv file, column 0, not a date time format:',
                  test_batch.iloc[0, 0])
