"""
Automated unit tests and mathematical parity validation for LUX Analytics.
Verifies consistency between the application calculations and the verified EDA notebook.
"""

import math
import unittest
import pandas as pd
import numpy as np

import data_loader
import analytics

class TestLuxAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load dataset once for tests
        cls.df = data_loader.load_dataset()

    def test_01_dataset_dimensions(self):
        """Verifies row count, column count, and temporal completeness."""
        self.assertEqual(len(self.df), 210240, "Total rows must equal exactly 210,240.")
        self.assertEqual(self.df['street_id'].nunique(), 24, "Grid must comprise exactly 24 street segments.")
        street_counts = self.df.groupby('street_id')['timestamp_local'].count()
        self.assertTrue((street_counts == 8760).all(), "Each street must have exactly 8,760 hourly records.")

    def test_02_feature_engineering_integrity(self):
        """Verifies derived physical and traffic features."""
        # Circuit rated power formula
        expected_kw = (self.df['luminaire_count'] * self.df['rated_watts_per_luminaire']) / 1000.0
        np.testing.assert_allclose(self.df['circuit_rated_kw'], expected_kw, rtol=1e-5)

        # Traffic period derivation
        self.assertTrue(self.df['traffic_period'].isin(['Morning Peak', 'Shoulder', 'Evening Peak', 'Deep Off-Peak']).all())

        # Valid population separation
        self.assertEqual(self.df['is_valid_energy'].sum(), 210114, "Valid energy subset must have 210,114 records (126 excluded).")

    def test_03_baseline_energy_benchmarks(self):
        """Verifies baseline energy and reference comparator against verified notebook values."""
        kpis = analytics.calculate_executive_kpis(self.df)
        self.assertAlmostEqual(kpis['valid_energy_kwh'], 195138.4808, delta=1.0)
        self.assertAlmostEqual(kpis['valid_full_kwh'], 234655.0620, delta=1.0)
        self.assertAlmostEqual(kpis['valid_savings_pct'], 16.8403, delta=0.05)

    def test_04_optimization_scenarios_parity(self):
        """Verifies scenario simulation energy hierarchy and notebook parity."""
        sim = analytics.simulate_scenarios(self.df)
        
        base_kwh = sim['base_total']
        full_kwh = sim['full_total']
        s1_kwh = sim['s1_kwh_total']
        s2_kwh = sim['s2_kwh_total']
        s3_kwh = sim['s3_kwh_total']

        # Hierarchy check: Full > Base > S1 > S2 > S3
        self.assertTrue(full_kwh > base_kwh > s1_kwh > s2_kwh > s3_kwh, "Energy hierarchy violation!")

        # Parity checks with verified EDA notebook
        self.assertAlmostEqual(s1_kwh, 182336.61, delta=10.0, msg="Scenario 1 energy mismatch")
        self.assertAlmostEqual(s2_kwh, 163519.50, delta=10.0, msg="Scenario 2 energy mismatch")
        self.assertAlmostEqual(s3_kwh, 158973.29, delta=10.0, msg="Scenario 3 energy mismatch")

    def test_05_zone_aggregation_parity(self):
        """Verifies zone summaries and total luminaires."""
        zone_sum = analytics.get_zone_summary(self.df)
        self.assertEqual(zone_sum['total_luminaires'].sum(), 447, "Total luminaires across 24 streets must equal 447.")
        self.assertEqual(len(zone_sum), 6, "Must contain all 6 zone types.")

if __name__ == "__main__":
    unittest.main()
