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


### Useage

- Supply pandas dataframe of features,  indexed by timeseries sampled by hour. 

````
    from models import *
    
    # create model with dataframe features, df, 
    # supply labels with dataframe with single column, df[['pm25]]
    base_model = BaseModel(df, df[['pm25']], look_back=3)

    # train model 
    base_model.train(nb_epoch=50)

    # test model, this will print RMSE errors
    testPredict = base_model.test()

    # get training and testing labels for measurement
    train_labels, test_labels = base_model.get_labels()
````

### Online system

Data pipeline is captured via indoor and outdoor data sources to influx db. 
- indoor sensor data
- outdoor atmosphere API

#### Pipeline
- Pull latest data into caching mechanism
- Feed cache data to model to display prediction
- At time interval, feed cache data to update model
- Pull next interval data and update cache

#### Monitoring
- grafana dashboard
- display current cache data (indoor, outdoor)
- display current prediction
- track historical predictions vs actual 

#### To be explored
- How much training data is needed before model is accurate?
- Add additional weather data
- Measure accuracy of t+N hours forecasting
