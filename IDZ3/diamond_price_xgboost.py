import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv(r'C:\IDZ3\diamonds_train.csv')
test_data = pd.read_csv(r'C:\IDZ3\diamonds_test.csv')

test_ids = test_data['id'].copy()

def create_smart_features(df):
    df_fe = df.copy()
    
    df_fe['volume'] = df_fe['x'] * df_fe['y'] * df_fe['z']
    df_fe['surface_area'] = 2 * (df_fe['x']*df_fe['y'] + df_fe['x']*df_fe['z'] + df_fe['y']*df_fe['z'])
    df_fe['depth_table_ratio'] = df_fe['depth'] / df_fe['table']
    
    clarity_rank = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    cut_rank = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    color_rank = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    
    df_fe['clarity_score'] = df_fe['clarity'].map(clarity_rank)
    df_fe['cut_score'] = df_fe['cut'].map(cut_rank)
    df_fe['color_score'] = df_fe['color'].map(color_rank)
    
    df_fe['total_quality'] = df_fe['cut_score'] + df_fe['color_score'] + df_fe['clarity_score']
    df_fe['value_index'] = df_fe['carat'] * df_fe['total_quality']
    
    df_fe['is_large'] = (df_fe['carat'] > 1).astype(int)
    df_fe['is_premium_cut'] = (df_fe['cut'].isin(['Ideal', 'Premium'])).astype(int)
    df_fe['is_high_clarity'] = (df_fe['clarity'].isin(['IF', 'VVS1', 'VVS2'])).astype(int)
    
    return df_fe

train_fe = create_smart_features(train_data)
test_fe = create_smart_features(test_data)

feature_columns = [
    'carat', 'depth', 'table', 'x', 'y', 'z',
    'volume', 'surface_area', 'depth_table_ratio',
    'clarity_score', 'cut_score', 'color_score',
    'total_quality', 'value_index', 'is_large',
    'is_premium_cut', 'is_high_clarity'
]

X = train_fe[feature_columns]
y = train_fe['price']
X_test = test_fe[feature_columns]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

lgb_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(X_train, y_train)

val_pred = lgb_model.predict(X_val)

val_r2 = r2_score(y_val, val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print("Results:")
print(f"R2 = {val_r2:.4f}, RMSE = ${val_rmse:,.2f}")

final_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

test_predictions = final_model.predict(X_test)

submission = pd.DataFrame({
    'id': test_ids,
    'price': test_predictions
})

submission['price'] = submission['price'].clip(lower=0)

submission_file = 'lightgbm_smart_features.csv'
submission.to_csv(submission_file, index=False)

print(f"File created: {submission_file}")
print(f"Size: {submission.shape}")
print(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature importance:")
for i, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']}: {row['importance']:.2f}")
