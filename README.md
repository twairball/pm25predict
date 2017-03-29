# PM2.5 Predict

Prototype RNN approach to predicting PM2.5 time series

### General Approach:
- RNN with LSTM K nodes with N past nodes
- Train baseline model 
- Predict on Test set
- measure loss

#### Questions:
** How to setup cross-validation? 
** How to add weather data? (wind, humidity, air pressure)

### Create Dataset
- Sample hourly data
- Add column for time of day, day of week
- Save as bcolz arrays

