from cleaned_historical import BitcoinPricePredictor

if __name__ == "__main__":
    # Instantiate the predictor
    predictor = BitcoinPricePredictor("BTC-USD(1).csv")
    
    # Prepare the data
    predictor.prepare_data()
    
    # Build the model
    predictor.build_model()
    
    # Train the model
    predictor.train_model(epochs=50)
    
    # Evaluate the model
    predictor.evaluate_model()
    
    # Optionally, make predictions
    # predictions = predictor.predict(predictor.test_data)
    # print(predictions)
