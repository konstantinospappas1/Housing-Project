USE housing_db;

SELECT 
    prefarea,
    COUNT(*) AS count_houses,
    ROUND(AVG(price), 0) AS avg_price
FROM housing
GROUP BY prefarea
ORDER BY avg_price DESC;


SELECT 
    mainroad,
    airconditioning,
    COUNT(*) AS num_houses,
    ROUND(AVG(price), 2) AS avg_price
FROM housing
GROUP BY mainroad, airconditioning
ORDER BY avg_price DESC;



SELECT 
    stories,
    parking,
    COUNT(*) AS num_houses,
    ROUND(AVG(price), 0) AS avg_price
FROM housing
GROUP BY stories, parking
ORDER BY avg_price DESC;


SELECT 
    bedrooms,
    bathrooms,
    COUNT(*) AS num_houses,
    ROUND(AVG(price), 0) AS avg_price
FROM housing
GROUP BY bedrooms, bathrooms
ORDER BY bedrooms, bathrooms;




WITH house_stats AS (
    SELECT 
        bedrooms,
        bathrooms,
        mainroad,
        airconditioning,
        COUNT(*) AS total_houses,
        ROUND(AVG(price), 2) AS avg_price,
        ROUND(MIN(price), 2) AS min_price,
        ROUND(MAX(price), 2) AS max_price,
        ROUND(AVG(area), 2) AS avg_area
    FROM housing
    WHERE parking IS NOT NULL
    GROUP BY bedrooms, bathrooms, mainroad, airconditioning
),
price_zones AS (
    SELECT 
        *,
        CASE 
            WHEN avg_price < 4000000 THEN 'Low Price Zone'
            WHEN avg_price BETWEEN 4000000 AND 6000000 THEN 'Medium Price Zone'
            ELSE 'High Price Zone'
        END AS price_zone
    FROM house_stats
)
SELECT 
    bedrooms,
    bathrooms,
    mainroad,
    airconditioning,
    price_zone,
    total_houses,
    avg_price,
    avg_area
FROM price_zones
WHERE total_houses >= 2
ORDER BY avg_price DESC, bedrooms ASC;
