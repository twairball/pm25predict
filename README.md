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


### Experimentation
- Supply pandas dataframe of features,  indexed by timeseries sampled by hour. 

````
    from pm25predict import ModelLoader, DatasetLoader, BaseModel
        
    # create dataset with dataframe features, df, 
    # supply labels with dataframe with single column, df[['pm25]]
    dataset_loader = DatasetLoader(df, df[['pm25']], look_back=3, ratio=0.9)

    # create model
    base_model = BaseModel(look_back=3)
    
    # train model 
    base_model.train(dataset_loader, nb_epoch=50)

    # test model on 10% of data, this will print RMSE errors
    testPredict = base_model.test(dataset_loader)

    # get training and testing labels for measurement
    train_labels, test_labels = dataset_loader.get_labels()
````

### Online prediction

Data pipeline is captured via indoor and outdoor data sources to influx db. 
- indoor sensor data
- outdoor atmosphere API

Cruncher will need to supply last 3 hours (look_back=3) data as Dataframe and create dataset. 

````
    from loaders import ModelLoader, InfluxDataLoader
    from models import BaseModel
    from datasets import Dataset, DatasetLoader

    # load data from influxdb
    df = InfluxDataLoader(database='gams', tables=['indoor', 'outdoor']).df

    # create dataset with dataframe features, df, 
    # supply labels with dataframe with single column, df[['pm25]]
    dataset_loader = DatasetLoader(df, df[['pm25']], look_back=3, ratio=1)

    # load current model
    model_path = "path/to/models/"
    loader = ModelLoader(model_path)

    # make prediction
    predict = loader.base_model.predict(dataset_loader)

    # update model
    loader.base_model.update(dataset_loader)

    # save model
    loader.save_model(model_path + "model_filename.h5")

````

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


### dev notes
Generate pip reqs
    
    pipreqs . --force
