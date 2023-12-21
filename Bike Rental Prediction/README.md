# predict how many bike rentals happen for a day
## Dataset:
Attribute Information:
- instant: record index
- dteday: date
- season: season (1: winter, 2: spring, 3: summer, 4: fall)
- yr: year (0: 2011, 1:2012)
- mnth: month (1 to 12)
- holiday: weather day is holiday or not
- weekday: day of the week
- workingday: if day is neither weekend nor holiday is 1, otherwise is 0.
- weathersit:
1. Clear,Fewclouds,Partlycloudy,Partlycloudy
2. Mist+Cloudy,Mist+Brokenclouds,Mist+Fewclouds,Mist
3. LightSnow,LightRain+Thunderstorm+Scatteredclouds,LightRain+Scatteredclouds 
4. HeavyRain+IcePallets+Thunderstorm+Mist,Snow+Fog
- temp: Normalized temperature in Celsius. The values are derived via (t-tmin)/(tmax-tmin), tmin=-8, t_max=+39 (only in hourly scale)
- atemp: Normalized feeling temperature in Celsius. The values are derived via (t-tmin)/(tmax-tmin), tmin=-16, t_max=+50 (only in hourly scale)
- hum: Normalized humidity. The values are divided to 100 (max)
- windspeed: Normalized wind speed. The values are divided to 67 (max)
- casual: count of casual users
- registered: count of registered users
- cnt: count of total rental bikes including both casual and registered (AV)
Tasks:
#### 1. Train and evaluate two different multiple regression models using different optimization algorithms. 
#### 2. Train and evaluate a polynomial regression model. Test at least two different models. 
#### 3. Select the machine learning model in task 2 and optimize it.
#### 4. Apply an ensemble learning technique to see whether this technique leads to a better performance.
