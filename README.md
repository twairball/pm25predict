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