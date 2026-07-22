-- =====================================================
-- Retail Sales Forecasting Database Schema
-- =====================================================

-- Drop tables if they already exist
DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS stores CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS oil_prices CASCADE;
DROP TABLE IF EXISTS holidays CASCADE;

---------------------------------------------------------
-- Stores Table
---------------------------------------------------------
CREATE TABLE stores (
    store_nbr INT PRIMARY KEY,
    city VARCHAR(50),
    state VARCHAR(50),
    store_type CHAR(1),
    cluster INT
);

---------------------------------------------------------
-- Sales Table
---------------------------------------------------------
CREATE TABLE sales (
    id INT PRIMARY KEY,,
    date DATE NOT NULL,
    store_nbr INT,
    family VARCHAR(100),
    sales NUMERIC(12,2),
    onpromotion INT,

    CONSTRAINT fk_store
        FOREIGN KEY(store_nbr)
        REFERENCES stores(store_nbr)
);

---------------------------------------------------------
-- Transactions Table
---------------------------------------------------------
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    date DATE,
    store_nbr INT,
    transactions INT,

    CONSTRAINT fk_transactions_store
        FOREIGN KEY(store_nbr)
        REFERENCES stores(store_nbr)
);

---------------------------------------------------------
-- Oil Prices Table
---------------------------------------------------------
CREATE TABLE oil_prices (
    date DATE PRIMARY KEY,
    dcoilwtico NUMERIC(10,2)
);

---------------------------------------------------------
-- Holidays Table
---------------------------------------------------------
CREATE TABLE holidays (
    id SERIAL PRIMARY KEY,
    date DATE,
    holiday_type VARCHAR(50),
    locale VARCHAR(30),
    locale_name VARCHAR(50),
    description VARCHAR(150),
    transferred BOOLEAN
);