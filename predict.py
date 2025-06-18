import pickle
import pandas as pd

pipeline = pickle.load(open('pipeline.pkl', 'rb'))

columns = ['Cuisines', 'Average Cost for two', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Price range', 'Rating color', 'Rating text', 'Votes']

test_input = pd.DataFrame([['French', '800','Botswana Pula(P)','No','Yes','No','Yes','4','Green','Very Good','300']], columns= columns)

predictions = pipeline.predict(test_input)
print('Rating', predictions)