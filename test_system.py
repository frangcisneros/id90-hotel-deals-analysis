# test_system.py
# Tests básicos para validar el sistema de Hotel Deals Classifier

import pytest
import pandas as pd
import auxiliary_functions
import config
from datetime import datetime


@pytest.fixture
def baselines_df(tmp_path):
    """Baselines mínimos escritos por el test (no dependen de outputs/ del repo).

    Contexto usado por los tests: destino 77 / enero / semana 2 con
    mean=150.0 y std=50.0, 120 observaciones y alta confianza.
    """
    baselines = pd.DataFrame({
        'destination_final': [77, 77],
        'destination_name': ['New York', 'New York'],
        'month': [1, 6],
        'week_in_month': [2, 2],
        'mean_price_std': [150.0, 180.0],
        'std_price_std': [50.0, 60.0],
        'min_price_std': [60.0, 70.0],
        'max_price_std': [320.0, 340.0],
        'count_obs': [120, 95],
        'low_confidence': [False, False],
    })

    baselines_path = tmp_path / "market_baselines.csv"
    baselines.to_csv(baselines_path, index=False)

    return auxiliary_functions.load_baselines(baselines_path)


class TestStandardization:
    """Tests de estandarización de precios."""
    
    def test_calcular_total_std_basic(self):
        """Test básico de estandarización."""
        row = {
            'avg_price_average': 450.0,
            'nights': 2,
            'number_of_rooms': 1,
            'number_of_adults': 2,
            'number_of_kids': 0
        }
        
        result = auxiliary_functions.calculate_price_std(
            pd.Series(row), 'avg_price_average'
        )
        
        # 450 / (2 * 1 * 2) = 112.5
        assert result == 112.5
    
    def test_calcular_total_std_with_kids(self):
        """Test con niños."""
        row = {
            'avg_price_average': 600.0,
            'nights': 3,
            'number_of_rooms': 2,
            'number_of_adults': 2,
            'number_of_kids': 1
        }
        
        result = auxiliary_functions.calculate_price_std(
            pd.Series(row), 'avg_price_average'
        )
        
        # 600 / (3 * 2 * 3) = 33.33
        assert abs(result - 33.33) < 0.01
    
    def test_calcular_total_std_multiple_rooms(self):
        """Test con múltiples habitaciones."""
        row = {
            'avg_price_average': 800.0,
            'nights': 4,
            'number_of_rooms': 2,
            'number_of_adults': 3,
            'number_of_kids': 1
        }
        
        result = auxiliary_functions.calculate_price_std(
            pd.Series(row), 'avg_price_average'
        )
        
        # 800 / (4 * 2 * 4) = 25.0
        assert result == 25.0


class TestClassification:
    """Tests de clasificación."""
    
    def test_classify_deal(self):
        """Test clasificación Deal."""
        z_score = -1.5
        classification = auxiliary_functions.classify_deal(z_score)
        assert classification == config.CLASSIFICATION_LABELS['deal']
    
    def test_classify_good_price(self):
        """Test clasificación Buen Precio."""
        z_score = -0.7
        classification = auxiliary_functions.classify_deal(z_score)
        assert classification == config.CLASSIFICATION_LABELS['good_price']
    
    def test_classify_normal(self):
        """Test clasificación Normal."""
        z_score = 0.2
        classification = auxiliary_functions.classify_deal(z_score)
        assert classification == config.CLASSIFICATION_LABELS['normal']
    
    def test_classify_expensive(self):
        """Test clasificación Caro."""
        z_score = 0.8
        classification = auxiliary_functions.classify_deal(z_score)
        assert classification == config.CLASSIFICATION_LABELS['expensive']
    
    def test_classify_boundary_deal(self):
        """Test en el límite de Deal (umbral exclusivo: z < -1.0)."""
        z_score = -1.0
        classification = auxiliary_functions.classify_deal(z_score)
        assert classification == config.CLASSIFICATION_LABELS['good_price']
    
    def test_classify_boundary_good(self):
        """Test en el límite de Buen Precio (umbral exclusivo: z < -0.5)."""
        z_score = -0.5
        classification = auxiliary_functions.classify_deal(z_score)
        assert classification == config.CLASSIFICATION_LABELS['normal']


class TestValidation:
    """Tests de validación de datos."""
    
    def test_validate_data_all_valid(self):
        """Test con todos los registros válidos."""
        df = pd.DataFrame({
            'nights': [1, 2, 3],
            'number_of_rooms': [1, 1, 2],
            'number_of_adults': [2, 2, 3],
            'number_of_kids': [0, 1, 0]
        })
        
        result = auxiliary_functions.validate_data(df)
        assert len(result) == 3
    
    def test_validate_data_filter_zero_nights(self):
        """Test filtrado de registros con nights=0."""
        df = pd.DataFrame({
            'nights': [1, 0, 3],  # 1 inválido
            'number_of_rooms': [1, 1, 2],
            'number_of_adults': [2, 2, 3],
            'number_of_kids': [0, 1, 0]
        })
        
        result = auxiliary_functions.validate_data(df)
        assert len(result) == 2  # Solo 2 válidos
    
    def test_validate_data_filter_zero_rooms(self):
        """Test filtrado de registros con rooms=0."""
        df = pd.DataFrame({
            'nights': [1, 2, 3],
            'number_of_rooms': [1, 0, 2],  # 1 inválido
            'number_of_adults': [2, 2, 3],
            'number_of_kids': [0, 1, 0]
        })
        
        result = auxiliary_functions.validate_data(df)
        assert len(result) == 2
    
    def test_validate_data_filter_zero_people(self):
        """Test filtrado de registros con adults+kids=0."""
        df = pd.DataFrame({
            'nights': [1, 2, 3],
            'number_of_rooms': [1, 1, 2],
            'number_of_adults': [2, 0, 3],  # Segundo registro inválido
            'number_of_kids': [0, 0, 0]
        })
        
        result = auxiliary_functions.validate_data(df)
        assert len(result) == 2


class TestBaselines:
    """Tests de carga de baselines."""
    
    def test_load_baselines(self, baselines_df):
        """Test carga de baselines."""
        baselines = baselines_df
        
        assert baselines is not None
        assert len(baselines) > 0
        assert 'destination_final' in baselines.columns
        assert 'mean_price_std' in baselines.columns
        assert 'std_price_std' in baselines.columns
    
    def test_baselines_structure(self, baselines_df):
        """Test estructura de baselines."""
        baselines = baselines_df
        
        required_cols = [
            'destination_final', 'month', 'week_in_month',
            'mean_price_std', 'std_price_std', 'count_obs', 'low_confidence'
        ]
        
        for col in required_cols:
            assert col in baselines.columns, f"Falta columna {col}"
    
    def test_baselines_all_high_confidence(self, baselines_df):
        """Test que todos los baselines sean de alta confianza."""
        baselines = baselines_df
        
        low_conf_count = baselines['low_confidence'].sum()
        assert low_conf_count == 0, "Todos los baselines deberían ser alta confianza"


class TestEvaluation:
    """Tests de evaluación de precios."""
    
    def test_evaluate_nyc_deal(self, baselines_df):
        """Test evaluación de un deal en NYC."""
        baselines = baselines_df
        
        # NYC, Enero, Semana 2, precio bajo
        result = auxiliary_functions.evaluate_hotel_price(
            destination_final=77,
            month=1,
            week_in_month=2,
            price_std=75.0,  # Muy bajo
            baselines_df=baselines
        )
        
        assert result['classification'] == config.CLASSIFICATION_LABELS['deal']
        assert result['z_score'] < -1.0
        assert result['confidence'] == 'high'
    
    def test_evaluate_nyc_expensive(self, baselines_df):
        """Test evaluación de precio caro en NYC."""
        baselines = baselines_df
        
        # NYC, Enero, Semana 2, precio alto
        result = auxiliary_functions.evaluate_hotel_price(
            destination_final=77,
            month=1,
            week_in_month=2,
            price_std=300.0,  # Muy alto
            baselines_df=baselines
        )
        
        assert result['z_score'] > 0.5
        assert result['confidence'] == 'high'
    
    def test_evaluate_no_baseline(self, baselines_df):
        """Test cuando no hay baseline disponible."""
        baselines = baselines_df
        
        # Contexto inexistente
        result = auxiliary_functions.evaluate_hotel_price(
            destination_final=99999,
            month=13,  # Mes inválido
            week_in_month=5,
            price_std=100.0,
            baselines_df=baselines
        )
        
        assert result['classification'] == config.CLASSIFICATION_LABELS['insufficient_data']
        assert result['z_score'] is None
        assert result['baseline_info'] is None


class TestConfig:
    """Tests de configuración."""
    
    def test_thresholds_exist(self):
        """Test que los thresholds estén configurados."""
        assert 'deal' in config.THRESHOLDS
        assert 'good_price' in config.THRESHOLDS
        assert 'normal_upper' in config.THRESHOLDS
    
    def test_thresholds_values(self):
        """Test valores de thresholds."""
        assert config.THRESHOLDS['deal'] == -1.0
        assert config.THRESHOLDS['good_price'] == -0.5
        assert config.THRESHOLDS['normal_upper'] == 0.5
    
    def test_classification_labels_exist(self):
        """Test que las etiquetas de clasificación existan."""
        assert 'deal' in config.CLASSIFICATION_LABELS
        assert 'good_price' in config.CLASSIFICATION_LABELS
        assert 'normal' in config.CLASSIFICATION_LABELS
        assert 'expensive' in config.CLASSIFICATION_LABELS


class TestWeekCalculation:
    """Tests de cálculo de semana del mes."""
    
    def test_get_week_in_month(self):
        """Test cálculo de semana."""
        assert config.get_week_in_month(1) == 1
        assert config.get_week_in_month(7) == 1
        assert config.get_week_in_month(8) == 2
        assert config.get_week_in_month(15) == 2
        assert config.get_week_in_month(16) == 3
        assert config.get_week_in_month(22) == 3
        assert config.get_week_in_month(23) == 4
        assert config.get_week_in_month(31) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
