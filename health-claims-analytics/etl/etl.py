import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from db import get_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "medical_insurance.csv")
SQL_PATH = os.path.join(BASE_DIR, "sql", "create_tables.sql")

def run_sql_file(conn, filepath):
    print(f"Running SQL schema file: {filepath}")
    with open(filepath, 'r') as f:
        sql = f.read()
    conn.execute(text(sql))
    print("SQL schema executed successfully.")

def data_quality_check(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting data quality checks...")
    missing_per_col = df.isnull().mean()
    if missing_per_col.any():
        print("Missing values per column:")
        print(missing_per_col[missing_per_col > 0])
    else:
        print("No missing values detected.")
    
    if 'person_id' not in df.columns:
        raise KeyError("Critical column 'person_id' is missing from data!")
    
    df = df.dropna(subset=['person_id'])
    return df

def clean_transform(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning and transforming data...")

    # Fill numeric columns missing values with median
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Fill categorical columns missing values with mode
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])
            else:
                df[col] = df[col].fillna("Unknown")

    # Add missing lifestyle columns with default values (optional)
    required_lifestyle_cols = ['exercise_frequency', 'sleep_hours', 'stress_level']
    for col in required_lifestyle_cols:
        if col not in df.columns:
            print(f"Column '{col}' missing, adding default values.")
            df[col] = 0

    # Normalize 'sex' column (capitalize)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.capitalize()

    # Map smoker to int 0/1
    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].astype(str).str.lower().map({'yes':1, 'no':0, 'never':0}).fillna(0).astype(int)

    # Ensure boolean columns are integer 0/1
    boolean_cols = [
        'hypertension', 'diabetes', 'copd', 'cardiovascular_disease', 'cancer_history',
        'kidney_disease', 'liver_disease', 'arthritis', 'mental_health', 'is_high_risk', 'had_major_procedure'
    ]
    for col in boolean_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df

def load_to_postgres(df: pd.DataFrame):
    engine = get_engine()
    with engine.begin() as conn:
        run_sql_file(conn, SQL_PATH)

        print("Loading data into staging_claims table...")
        df.to_sql('staging_claims', conn, if_exists='replace', index=False)

        # Upsert dim_person
        print("Upserting dim_person...")
        conn.execute(text("""
            INSERT INTO dim_person (
                person_id, age, sex, region, urban_rural, income, education, marital_status, 
                employment_status, household_size, dependents
            )
            SELECT DISTINCT person_id, age, sex, region, urban_rural, income, education, marital_status, 
                employment_status, household_size, dependents
            FROM staging_claims
            ON CONFLICT (person_id) DO UPDATE SET
                age = EXCLUDED.age,
                sex = EXCLUDED.sex,
                region = EXCLUDED.region,
                urban_rural = EXCLUDED.urban_rural,
                income = EXCLUDED.income,
                education = EXCLUDED.education,
                marital_status = EXCLUDED.marital_status,
                employment_status = EXCLUDED.employment_status,
                household_size = EXCLUDED.household_size,
                dependents = EXCLUDED.dependents;
        """))

        # Upsert dim_lifestyle
        print("Upserting dim_lifestyle...")
        conn.execute(text("""
            INSERT INTO dim_lifestyle (
                person_id, bmi, smoker, alcohol_freq, exercise_frequency, sleep_hours, stress_level
            )
            SELECT DISTINCT
                person_id,
                bmi,
                CASE WHEN smoker = 1 THEN TRUE ELSE FALSE END,
                alcohol_freq,
                exercise_frequency,
                sleep_hours,
                stress_level
            FROM staging_claims
            ON CONFLICT (person_id) DO UPDATE SET
                bmi = EXCLUDED.bmi,
                smoker = EXCLUDED.smoker,
                alcohol_freq = EXCLUDED.alcohol_freq,
                exercise_frequency = EXCLUDED.exercise_frequency,
                sleep_hours = EXCLUDED.sleep_hours,
                stress_level = EXCLUDED.stress_level;
        """))

        # Upsert dim_health_conditions
        print("Upserting dim_health_conditions...")
        conn.execute(text("""
            INSERT INTO dim_health_conditions (
                person_id, hypertension, diabetes, copd, cardiovascular_disease, cancer_history,
                kidney_disease, liver_disease, arthritis, mental_health, chronic_count,
                systolic_bp, diastolic_bp, ldl, hba1c, risk_score, is_high_risk
            )
            SELECT DISTINCT
                person_id,
                CASE WHEN hypertension = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN diabetes = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN copd = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN cardiovascular_disease = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN cancer_history = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN kidney_disease = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN liver_disease = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN arthritis = 1 THEN TRUE ELSE FALSE END,
                CASE WHEN mental_health = 1 THEN TRUE ELSE FALSE END,
                chronic_count,
                systolic_bp,
                diastolic_bp,
                ldl,
                hba1c,
                risk_score,
                CASE WHEN is_high_risk = 1 THEN TRUE ELSE FALSE END
            FROM staging_claims
            ON CONFLICT (person_id) DO UPDATE SET
                hypertension = EXCLUDED.hypertension,
                diabetes = EXCLUDED.diabetes,
                copd = EXCLUDED.copd,
                cardiovascular_disease = EXCLUDED.cardiovascular_disease,
                cancer_history = EXCLUDED.cancer_history,
                kidney_disease = EXCLUDED.kidney_disease,
                liver_disease = EXCLUDED.liver_disease,
                arthritis = EXCLUDED.arthritis,
                mental_health = EXCLUDED.mental_health,
                chronic_count = EXCLUDED.chronic_count,
                systolic_bp = EXCLUDED.systolic_bp,
                diastolic_bp = EXCLUDED.diastolic_bp,
                ldl = EXCLUDED.ldl,
                hba1c = EXCLUDED.hba1c,
                risk_score = EXCLUDED.risk_score,
                is_high_risk = EXCLUDED.is_high_risk;
        """))

        # Upsert dim_healthcare_utilization
        print("Upserting dim_healthcare_utilization...")
        conn.execute(text("""
            INSERT INTO dim_healthcare_utilization (
                person_id, visits_last_year, hospitalizations_last_3yrs, days_hospitalized_last_3yrs,
                medication_count, proc_imaging, proc_surgery, proc_psycho, proc_consult_count, proc_lab, had_major
            )
            SELECT DISTINCT
                person_id,
                visits_last_year,
                hospitalizations_last_3yrs,
                days_hospitalized_last_3yrs,
                medication_count,
                proc_imaging_count,
                proc_surgery_count,
                proc_physio_count,
                proc_consult_count,
                proc_lab_count,
                CASE WHEN had_major_procedure = 1 THEN TRUE ELSE FALSE END
            FROM staging_claims
            ON CONFLICT (person_id) DO UPDATE SET
                visits_last_year = EXCLUDED.visits_last_year,
                hospitalizations_last_3yrs = EXCLUDED.hospitalizations_last_3yrs,
                days_hospitalized_last_3yrs = EXCLUDED.days_hospitalized_last_3yrs,
                medication_count = EXCLUDED.medication_count,
                proc_imaging = EXCLUDED.proc_imaging,
                proc_surgery = EXCLUDED.proc_surgery,
                proc_psycho = EXCLUDED.proc_psycho,
                proc_consult_count = EXCLUDED.proc_consult_count,
                proc_lab = EXCLUDED.proc_lab,
                had_major = EXCLUDED.had_major;
        """))

        # Upsert dim_insurance_policy
        print("Upserting dim_insurance_policy...")
        conn.execute(text("""
            INSERT INTO dim_insurance_policy (
                person_id, plan_type, network_tier, deductible, copay, policy_term_years,
                policy_changes_last_2yrs, provider_quality
            )
            SELECT DISTINCT
                person_id, plan_type, network_tier, deductible, copay, policy_term_years,
                policy_changes_last_2yrs, provider_quality
            FROM staging_claims
            ON CONFLICT (person_id) DO UPDATE SET
                plan_type = EXCLUDED.plan_type,
                network_tier = EXCLUDED.network_tier,
                deductible = EXCLUDED.deductible,
                copay = EXCLUDED.copay,
                policy_term_years = EXCLUDED.policy_term_years,
                policy_changes_last_2yrs = EXCLUDED.policy_changes_last_2yrs,
                provider_quality = EXCLUDED.provider_quality;
        """))

        print("Truncating fact_medical_costs_claims...")
        conn.execute(text("TRUNCATE TABLE fact_medical_costs_claims;"))

        print("Inserting into fact_medical_costs_claims...")
        conn.execute(text("""
            INSERT INTO fact_medical_costs_claims (
                person_id, annual_medical_cost, annual_premium, monthly_premium,
                claims_count, avg_claim_amount, total_claims_paid
            )
            SELECT
                person_id, annual_medical_cost, annual_premium, monthly_premium,
                claims_count, avg_claim_amount, total_claims_paid
            FROM staging_claims;
        """))

    print("Data load complete.")

if __name__ == "__main__":
    print("Loading CSV from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = data_quality_check(df)
    df_clean = clean_transform(df)
    load_to_postgres(df_clean)
