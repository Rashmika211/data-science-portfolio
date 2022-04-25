#Customers and transactions
SELECT * FROM sales.customers;
SELECT * FROM sales.transactions;

#total number of customers
SELECT count(*) as 'Total Customers' FROM customers;

#Transactions for Hyderabad market
SELECT * FROM sales.transactions where market_code='Mark014';

#Distinct product codes that were sold in Hyderabad
SELECT distinct product_code FROM sales.transactions where market_code='Mark014';

#Transactions where currency is not INR dollars
SELECT * from sales.transactions where currency!="INR";

#Transactions in 2019 
SELECT transactions.*, date.*
FROM
    transactions
INNER JOIN
    date ON transactions.order_date = date.date
WHERE
    date.year = 2019;

#Total revenue year-wise
SELECT 
    year(order_date) as Year,SUM(transactions.sales_amount) as 'Sales amount'
FROM
    transactions
GROUP BY year(order_date)
ORDER BY order_date,sales_amount;

#total revenue in year 2020, June Month
SELECT 
    year(order_date) as year,month(order_date) as month,SUM(transactions.sales_amount) as 'Sales amount'
FROM
    transactions
        INNER JOIN
    date ON transactions.order_date = date.date
WHERE
    date.year = 2020 and date.month_name="June";

#Show total revenue in year 2020 in Hyderabad
SELECT 
    SUM(transactions.sales_amount) as 'Sales amount'
FROM
    transactions
        INNER JOIN
    date ON transactions.order_date = date.date
WHERE
    date.year = 2020
        AND transactions.market_code = 'Mark014';

#Total profit year-wise
SELECT 
    year(order_date) as year,SUM(transactions.profit) as 'profit'
FROM
    transactions
GROUP BY year(order_date)
ORDER BY order_date,profit;

#total profit in year 2020 in Hyderabad
SELECT 
    year(order_date) as year,SUM(transactions.profit) as 'profit'
FROM
    transactions
WHERE
    year(order_date) = 2020
        AND transactions.market_code = 'Mark014';
        
#Total profit month-wise in 2019
SELECT 
    MONTH(order_date) as month, SUM(transactions.profit) as 'profit'
FROM
    transactions
WHERE
    YEAR(order_date) = 2019
GROUP BY MONTH(order_date)
ORDER BY MONTH(order_date) , profit;
        
#Total profit-margin-percentage year-wise
SELECT 
    year(order_date) as year,SUM(transactions.profit_margin_percentage) as profit_margin_percentage
FROM
    transactions
GROUP BY year(order_date)
ORDER BY order_date,profit_margin_percentage;