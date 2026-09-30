-- Drop tables if they exist 
DROP TABLE IF EXISTS fact_medical_costs_claims CASCADE;
DROP TABLE IF EXISTS dim_insurance_policy CASCADE;
DROP TABLE IF EXISTS dim_healthcare_utilization CASCADE;
DROP TABLE IF EXISTS dim_health_conditions CASCADE;
DROP TABLE IF EXISTS dim_lifestyle CASCADE;
DROP TABLE IF EXISTS dim_person CASCADE;

-- dim_person table with demographics & socioeconomics
CREATE TABLE IF NOT EXISTS dim_person (
    person_id VARCHAR(50) PRIMARY KEY,
    age INT,
    sex VARCHAR(10),
    region VARCHAR(50),
    urban_rural VARCHAR(20),
    income NUMERIC(12,2),
    education VARCHAR(50),
    marital_status VARCHAR(50),
    employment_status VARCHAR(50),
    household_size INT,
    dependents INT
);

-- dim_lifestyle
CREATE TABLE IF NOT EXISTS dim_lifestyle (
    person_id VARCHAR(50) PRIMARY KEY REFERENCES dim_person(person_id),
    bmi NUMERIC(5,2),
    smoker BOOLEAN,
    alcohol_freq VARCHAR(50),
    exercise_frequency VARCHAR(50),
    sleep_hours NUMERIC(3,1),
    stress_level VARCHAR(50)
);

-- dim_health_conditions
CREATE TABLE IF NOT EXISTS dim_health_conditions (
    person_id VARCHAR(50) PRIMARY KEY REFERENCES dim_person(person_id),
    hypertension BOOLEAN,
    diabetes BOOLEAN,
    copd BOOLEAN,
    cardiovascular_disease BOOLEAN,
    cancer_history BOOLEAN,
    kidney_disease BOOLEAN,
    liver_disease BOOLEAN,
    arthritis BOOLEAN,
    mental_health BOOLEAN,
    chronic_count INT,
    systolic_bp INT,
    diastolic_bp INT,
    ldl NUMERIC(6,2),
    hba1c NUMERIC(4,2),
    risk_score NUMERIC(5,2),
    is_high_risk BOOLEAN
);

-- dim_healthcare_utilization
CREATE TABLE IF NOT EXISTS dim_healthcare_utilization (
    person_id VARCHAR(50) PRIMARY KEY REFERENCES dim_person(person_id),
    visits_last_year INT,
    hospitalizations_last_3yrs INT,
    days_hospitalized_last_3yrs INT,
    medication_count INT,
    proc_imaging INT,
    proc_surgery INT,
    proc_psycho INT,
    proc_consult_count INT,
    proc_lab INT,
    had_major BOOLEAN
);

-- dim_insurance_policy
CREATE TABLE IF NOT EXISTS dim_insurance_policy (
    person_id VARCHAR(50) PRIMARY KEY REFERENCES dim_person(person_id),
    plan_type VARCHAR(50),
    network_tier VARCHAR(50),
    deductible NUMERIC(12,2),
    copay NUMERIC(12,2),
    policy_term_years INT,
    policy_changes_last_2yrs INT,
    provider_quality VARCHAR(50)
);

-- fact_medical_costs_claims
CREATE TABLE IF NOT EXISTS fact_medical_costs_claims (
    person_id VARCHAR(50) PRIMARY KEY REFERENCES dim_person(person_id),
    annual_medical_cost NUMERIC(12,2),
    annual_premium NUMERIC(12,2),
    monthly_premium NUMERIC(12,2),
    claims_count INT,
    avg_claim_amount NUMERIC(12,2),
    total_claims_paid NUMERIC(12,2)
);

