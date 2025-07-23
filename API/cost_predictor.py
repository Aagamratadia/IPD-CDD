import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class CarDamageCostPredictor:
    def __init__(self):
        # Base cost by damage location
        self.location_cost = {
            'hood': 2000,
            'door': 1500,
            'bumper': 1800,
            'roof': 2200,
            'side_mirror': 800,
            'fender': 1600
        }
        self.locations = list(self.location_cost.keys())

        # Brand multiplier
        self.brand_multiplier = {
            'Maruti': 1.8,
            'Toyota': 2.0,
            'Honda': 2.05,
            'Hyundai': 2.95,
            'BMW': 40.8,
            'Mercedes': 50.0,
            'Ford': 5.1,
            'Kia': 1.9,
            'Audi': 40.9
        }
        self.brands = list(self.brand_multiplier.keys())

        # Severity multiplier
        self.severity_multiplier = {
            'minor': 1.0,
            'moderate': 1.5,
            'severe': 2.2
        }
        self.severities = list(self.severity_multiplier.keys())

        # Initialize encoders
        self.le_brand = LabelEncoder()
        self.le_location = LabelEncoder()
        self.le_severity = LabelEncoder()
        self.le_price = LabelEncoder()

        # Price range bins
        self.bins = list(range(0, 105, 5))
        self.labels = [f'{i}-{i+5}L' for i in range(0, 100, 5)]

        # Initialize model
        self.model = None
        self._generate_training_data()
        self._train_model()

    def _get_price_multiplier(self, price):
        if price < 10:
            return 0.9
        elif price < 15:
            return 1.0
        elif price < 20:
            return 1.1
        elif price < 25:
            return 1.2
        elif price < 30:
            return 1.3
        elif price < 35:
            return 1.4
        elif price < 40:
            return 1.5
        elif price < 45:
            return 1.6
        elif price < 50:
            return 1.7
        elif price < 55:
            return 1.8
        elif price < 60:
            return 1.9
        elif price < 65:
            return 2.0
        elif price < 70:
            return 2.1
        elif price < 75:
            return 2.2
        elif price < 80:
            return 2.3
        elif price < 85:
            return 2.4
        elif price < 90:
            return 2.5
        elif price < 95:
            return 2.6
        elif price < 100:
            return 2.7
        else:
            return 2.8

    def _generate_training_data(self):
        # Generate synthetic dataset
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'Brand': np.random.choice(self.brands, size=n_samples),
            'Scratch_Location': np.random.choice(self.locations, size=n_samples),
            'Severity': np.random.choice(self.severities, size=n_samples),
            'Car_Price_Lakhs': np.random.randint(5, 101, size=n_samples),
            'Damage_Area': np.random.uniform(0.01, 0.25, size=n_samples)  # Simulate 1% to 25% of image
        })

        # Calculate multipliers and base costs
        df['Brand_Multiplier'] = df['Brand'].map(self.brand_multiplier)
        df['Location_Base_Cost'] = df['Scratch_Location'].map(self.location_cost)
        df['Severity_Multiplier'] = df['Severity'].map(self.severity_multiplier)
        df['Price_Range'] = pd.cut(df['Car_Price_Lakhs'], bins=self.bins, labels=self.labels)
        df['Price_Multiplier'] = df['Car_Price_Lakhs'].apply(self._get_price_multiplier)

        # Calculate estimated cost (now includes damage area as a multiplier)
        df['Estimated_Cost'] = (
            df['Location_Base_Cost'] *
            df['Brand_Multiplier'] *
            df['Severity_Multiplier'] *
            df['Price_Multiplier'] *
            (1 + 2 * df['Damage_Area'])  # Area has a strong effect (1x to 3x)
        )

        # Encode categorical variables
        df['Brand_encoded'] = self.le_brand.fit_transform(df['Brand'])
        df['Location_encoded'] = self.le_location.fit_transform(df['Scratch_Location'])
        df['Severity_encoded'] = self.le_severity.fit_transform(df['Severity'])
        df['Price_encoded'] = self.le_price.fit_transform(df['Price_Range'])

        self.df = df

    def _train_model(self):
        X = self.df[['Brand_encoded', 'Location_encoded', 'Severity_encoded', 'Price_encoded', 'Damage_Area']]
        y = self.df['Estimated_Cost']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.mae = mean_absolute_error(y_test, y_pred)

    def predict_cost(self, brand, location, severity, car_price_lakhs, damage_area=None):
        try:
            # Validate inputs
            if brand not in self.brands:
                raise ValueError(f"Invalid brand. Must be one of: {self.brands}")
            if location not in self.locations:
                raise ValueError(f"Invalid location. Must be one of: {self.locations}")
            if severity not in self.severities:
                raise ValueError(f"Invalid severity. Must be one of: {self.severities}")
            if not (5 <= car_price_lakhs <= 100):
                raise ValueError("Car price must be between 5 and 100 lakhs")
            if damage_area is None:
                damage_area = 0.05  # Default to 5% if not provided

            # Encode inputs
            brand_code = self.le_brand.transform([brand])[0]
            location_code = self.le_location.transform([location])[0]
            severity_code = self.le_severity.transform([severity])[0]

            # Bin and encode price
            price_range = pd.cut([car_price_lakhs], bins=self.bins, labels=self.labels)[0]
            price_code = self.le_price.transform([price_range])[0]

            # Predict
            predicted_cost = self.model.predict([[brand_code, location_code, severity_code, price_code, damage_area]])[0]

            return {
                "estimated_cost": round(predicted_cost, 2),
                "brand": brand,
                "location": location,
                "severity": severity,
                "car_price_lakhs": car_price_lakhs,
                "damage_area": damage_area,
                "model_mae": round(self.mae, 2)
            }

        except Exception as e:
            raise ValueError(f"Error in prediction: {str(e)}")

# Create a singleton instance
cost_predictor = CarDamageCostPredictor() 