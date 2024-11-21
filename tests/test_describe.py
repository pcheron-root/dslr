import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from scripts.describe import Describe
import statistics

casesList = [
    ("valid", [1.0, -5.344231, 6.53, 1000], None),
    ("valid", [0.0, 0.0, 0.0, 0.0], None),
    ("empty", [], Exception),
    ("no data", None, TypeError),
]


class TestDescribe:
    def setup_method(self):
        self.describe = Describe()

    def cases(function):
        return [
            ("valid", [1.0, -1.0, 2.0, 3.0], function([1.0, -1.0, 2.0, 3.0]), None),
            ("valid", [0.0, 0.0, 0.0, 0.0], function([0.0, 0.0, 0.0, 0.0]), None),
            ("empty", [], None, Exception),
            ("no data", None, None, TypeError),
        ]

    def apply_tests(self, func, expected_func, case_type, data, expected_exception):
        if case_type == "valid":
            expected_output = expected_func(data)
            result = func(data)
            assert result == expected_output
        else:
            with pytest.raises(expected_exception):
                func(data)

    @pytest.mark.parametrize("case_type, data, expected_exception", casesList)
    def test_min(self, case_type, data, expected_exception):
        self.apply_tests(self.describe.min, min, case_type, data, expected_exception)

    @pytest.mark.parametrize("case_type, data, expected_exception", casesList)
    def test_max(self, case_type, data, expected_exception):
        self.apply_tests(self.describe.max, max, case_type, data, expected_exception)

    @pytest.mark.parametrize("case_type, data, expected_exception", casesList)
    def test_mean(self, case_type, data, expected_exception):
        self.apply_tests(
            self.describe.mean, statistics.mean, case_type, data, expected_exception
        )

    @pytest.mark.parametrize(
        "case_type, data, expected_output, expected_exception",
        cases(statistics.quantiles),
    )
    def test_quantiles(self, case_type, data, expected_output, expected_exception):
        describe = Describe()

        if case_type == "valid":
            result = [
                describe.find_percentile(data, 25),
                describe.find_percentile(data, 50),
                describe.find_percentile(data, 75),
            ]
            assert result == expected_output
        else:
            with pytest.raises(expected_exception):
                describe.find_percentile(data, None)
