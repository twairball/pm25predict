# PM2.5 Predict

Prototype RNN approach to predicting PM2.5 time series

### Experimentation
- Supply pandas dataframe of features,  indexed by timeseries sampled by hour. 

````
    from pm25predict import ModelLoader, DatasetLoader, ModelContext
        
    # load data from influxdb
    df = InfluxDataLoader(database='gams', tables=['indoor', 'outdoor']).df

    # create dataset with dataframe features, df, 
    # supply labels with dataframe with single column, df[['pm25]]
    dataset_loader = DatasetLoader(df, df[['pm25']], look_back=3, ratio=0.9)

    # create model
    model_context = ModelContext(look_back=3)
    
    # train model 
    model_context.train(dataset_loader, nb_epoch=50)

    # test model on 10% of data, this will print RMSE errors
    testPredict = model_context.test(dataset_loader)

    # get training and testing labels for measurement
    train_labels, test_labels = dataset_loader.get_labels()
````

### Online prediction

Data pipeline is captured via indoor and outdoor data sources to influx db. 
- indoor sensor data
- outdoor atmosphere API

Cruncher will need to supply last 3+2 hours (look_back=3) data as Dataframe and create dataset. 

````
    from pm25predict import ModelLoader, DatasetLoader, ModelContext

    # load dataset with ratio = 0.0 for all testing features
    dataset_loader = DatasetLoader(df, df[['pm25']], look_back=3, ratio=0.)

    # load current model
    model_path = "path/to/models/"
    loader = ModelLoader(model_path)

    # make prediction
    predict = loader.model_context.predict(dataset_loader)
````

````
    # load dataset with ratio = 1.0 for all training features
    dataset_loader = DatasetLoader(df, df[['pm25']], look_back=3, ratio=1.)

    # update model
    loader.model_context.update(dataset_loader)

    # save model
    loader.save_model()
````

### dev notes
Generate pip reqs
    
    pipreqs . --force

#### TODO
- Monitoring with grafana dashboard, and measure predictions vs actual
- Add weather data (wind, humidity, air pressure)
- Predict t+N hours forecasting, and measure accuracy. 

