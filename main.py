import pandas as pd
import multiprocessing as mp
import pickle
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect
from gensim.models.word2vec import Word2Vec
from summarizer import Summarizer



def preprocessing():
    # Breaking down the large 7Gb dataset into chunks of 1_000_000 rows each
    chunk_size=1_000_000
    batch_no=1

    for chunk in pd.read_json('./yelp_academic_dataset_review.json', lines=True, chunksize=chunk_size):
        #sorting all the reviews according to date
        chunk['date'] = pd.to_datetime(chunk['date'])
        chunk = chunk.sort_values('date')
        chunk.reset_index(inplace=True, drop=True)
        chunk.to_csv('./chunk_raw_'+str(batch_no)+'.csv',index=False)
        print(f'Chunk {batch_no} complete')
        batch_no+=1
    print('Chunking complete')
    print('*************************')
    # the code was looked up at https://towardsdatascience.com/loading-large-datasets-in-pandas-11bdddd36f7b

    # creating a dataset of businesses that are restaurants
    print('Restaurants dataset creation start')
    df_business = pd.read_json('./yelp_academic_dataset_business.json', lines=True)
    df_business.dropna(inplace=True)
    df_rest = df_business.copy()
    df_rest = df_rest[df_rest['categories'].str.contains('Restaurant')]
    df_rest.dropna(inplace=True)
    df_rest.to_csv('./restaurants.csv', index=False)
    print('Restaurants dataset complete')
    print('*************************')


def only_rest():
    # deleting all the reviews that are not about restaurants
    df_rest = pd.read_csv('./restaurants.csv')
    bus_id = list(df_rest['business_id'])
    for i in range(1, 10):
        print(f'Iteration {i} in 9 ({round(i/9, 2)})')
        df=pd.read_csv(f'./chunk_raw_{i}.csv')
        df_rr = pd.DataFrame([df.loc[i,:] for i in range(len(df['business_id'])) if df.loc[i, 'business_id'] in bus_id])
        df_rr.to_csv(f'./chunk{i}.csv', index=False)
    print('Deleting non-restaurant reviews complete')    
    print('Deleting all non-restaurant reviews FINISH')
    print('*************************')


def more_details():
    #adding info about business to the reviews dataframe
    print('Adding restaurant details to reviews dataset START')
    df_rest = pd.read_csv('./restaurants.csv')

    for i in range(1, 10):    
        df = pd.read_csv(f'./chunk{i}.csv')
        all_bus = []
        the_list = list(df['business_id'])

        for bus_id in the_list:
            business = {}  
            need = df_rest[df_rest['business_id'] == bus_id]
            need.reset_index(inplace=True, drop=True)
            business['bus_id'] = need['business_id'][0]
            business['name'] = need['name'][0]
            business['address'] = need['address'][0]
            business['city'] = need['city'][0]
            business['state'] = need['state'][0]
            business['postal_code'] = need['postal_code'][0]
            business['latitude'] = need['latitude'][0]
            business['longitude'] = need['longitude'][0]
            all_bus.append(business)

        bus_id = []
        name = []
        address = []
        city = []
        state = []
        postal_code = []
        latitude = []
        longitude = []

        for z in all_bus:
            bus_id.append(z['bus_id'])
            name.append(z['name'])
            address.append(z['address'])
            city.append(z['city'])
            state.append(z['state'])
            postal_code.append(z['postal_code'])
            latitude.append(z['latitude'])
            longitude.append(z['longitude'])

        df['name'] = name
        df['address'] = address
        df['city'] = city
        df['state'] = state
        df['postal_code'] = postal_code
        df['latitude'] = latitude
        df['longitude'] = longitude
        df.to_csv(f'./final_chunk{i}.csv', index=False)
        print(f'Complete part {i} of 9')
    print('Adding restaurant details to reviews dataset END')
    print('*************************')


def combine():
    #combining all the chunks into single dataframe
    print('combine all chunks into single dataset')
    df = pd.DataFrame()
    for i in range(1, 10):
        chunk = pd.read_csv(f'./final_chunk{i}.csv')
        df = df.append(chunk)
        print(f'Chunk {i} merged')
    df.to_csv('./final_all_reviews.csv', index=False)
    print('Combine all chunks into single dataset complete')
    print('*************************')


def time_clean(time):
    #Deleting all the reviews from datasat that are older then {time}
    print(f'Time cleaning from {time} started')
    df = pd.read_csv('./final_all_reviews.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= time]
    df.to_csv('./recent_reviews.csv', index=False)
    print(f'Time cleaning from {time} finished')
    print('*************************')


def rearrange():
    #rearranging the dataset columns
    print('Rearranging and cleaning of columns start')
    df = pd.read_csv('./recent_reviews.csv')
    needed_info = ['date', 'business_id', 'bus_name', 'address', 'city', 'state',
                   'postal_code', 'latitude', 'longitude', 'stars',
                   'funny', 'cool', 'text']
    df = df[needed_info]
    df.to_csv('./rearranged_reviews.csv', index=False)
    print('Rearrange and cleaning complete')
    print('*************************')


def lang_clear():
    #Removing all the non-english reviews
    print('Cleaning from non-english language reviews start')
    df = pd.read_csv('./rearranged_reviews.csv')
    for i in range(df.shape[0]):
        try:
            if not detect(df.loc[i,'text'])=='en':
                df.drop(i, inplace=True)
        except:
            df.drop(i, inplace=True)
    df.dropna(inplace=True)
    df.to_csv('./eng_reviews.csv', index=False)
    print('Language cleaning COMPLETE')
    print('*************************')


def us_only(j):
    #leaving only US states and cities
    if not df.loc[j, 'state'] in states:
        df.drop(j, inplace=True)

# in case there is a need to work on one city
# def choose_city_state(city, state):
#     #Limiting the dataset to single city as the whole dataset is too large
#     print('Choosing city and state')
#     df_final = pd.read_csv('./final_reviews.csv')
#     df_business = pd.read_csv('./restaurants.csv')
#     df_city_bus = df_business[(df_business['city'] == city) & (df_business['state'] == state)].copy()
#     df_city_rev = df_final[(df_final['city']==city)&(df_final['state']==state)].copy()
#     df_city_bus.to_csv('./city_restaurants.csv', index=False)
#     df_city_rev.to_csv('./city_reviews.csv', index=False)
#     print('Choosing city and state complete')


def model_token_lemm():
    #tokenizing and lemmatizing the reviews text data
    print('Loading data')
    
    # Code for standalone city processing
    # df_final = pd.read_csv('./city_reviews.csv')
    # df_business = pd.read_csv('./city_restaurants.csv')

    # General process code
    df_final = pd.read_csv('./final_reviews.csv')
    df_business = pd.read_csv('./restaurants.csv')

    r_tokenizer = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')
    lemmatizer = WordNetLemmatizer()
    
    df_business['tokens'] = ['nothing' for _ in range(df_business.shape[0])]
    print('Starting tokenization and lemmatizing')
    counter = 1
    
    for business in df_business['business_id']:
        bus_tokens = []
        text = df_final[df_final['business_id']==business]['text']
        bus_tokens = []
        
        for t in text:
            tokens = r_tokenizer.tokenize(t)
            for word in tokens:
                if not word in stopwords.words('english'):
                    bus_tokens.append(word.lower())
        
        bus_lemmas = [lemmatizer.lemmatize(word) for word in bus_tokens]
        bus_lemmas = ' '.join(bus_lemmas)
        the_index = df_business[df_business['business_id'] == business].index
        df_business.loc[the_index, 'tokens'] = bus_lemmas
        print(f'Finished business {counter} out of {df_business.shape[0]}.')
        counter += 1
    df_business.to_csv('./city_restaurants.csv', index=False)


def dimensions():
    #Loading dimensions from the csv file and evaluating the lemmatized text
    print('Loading data set')
    df_restaurants = pd.read_csv('./city_restaurants.csv')
    print('Loading dimensions for food quality, service speed, service quality and atmosphere') 
    df_restaurants.drop(columns='Unnamed: 0', inplace=True)
    dimensions = pd.read_csv('./dimensions.csv')
    
    lemmatizer = WordNetLemmatizer()
    
    foodq_plus = [lemmatizer.lemmatize(word) for word in list(dimensions['foodq_plus']) if not word == 'empty']
    foodq_minus = [lemmatizer.lemmatize(word) for word in list(dimensions['foodq_minus']) if not word == 'empty']
    services_plus = [lemmatizer.lemmatize(word) for word in list(dimensions['services_plus']) if not word == 'empty']
    services_minus = [lemmatizer.lemmatize(word) for word in list(dimensions['services_minus']) if not word == 'empty']
    serviceq_plus = [lemmatizer.lemmatize(word) for word in list(dimensions['serviceq_plus']) if not word == 'empty']
    serviceq_minus = [lemmatizer.lemmatize(word) for word in list(dimensions['serviceq_minus']) if not word == 'empty']
    atmosphere_plus = [lemmatizer.lemmatize(word) for word in list(dimensions['atmosphere_plus']) if not word == 'empty']
    atmosphere_minus = [lemmatizer.lemmatize(word) for word in list(dimensions['atmosphere_minus']) if not word == 'empty']

    #creating columns for counting the ratings
    df_restaurants['fp'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['fm'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['ssp'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['ssm'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['sqp'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['sqm'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['ap'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]
    df_restaurants['am'] = ['rating goes here' for _ in range(df_restaurants.shape[0])]

    print('Calculating the dimensions')
    counter = 1
    for i in range(df_restaurants.shape[0]):
        rest_fp, rest_fm, rest_ssp, rest_ssm, rest_sqp, rest_sqm, rest_ap, rest_am = 0, 0, 0, 0, 0, 0, 0, 0
        try:
            for word in df_restaurants.loc[i, 'tokens'].split():
                if word in foodq_plus:
                    rest_fp +=1
                if word in foodq_minus:
                    rest_fm +=1
                if word in services_plus:
                    rest_ssp +=1
                if word in services_minus:
                    rest_ssm +=1
                if word in serviceq_plus:
                    rest_sqp +=1
                if word in serviceq_minus:
                    rest_sqm +=1
                if word in atmosphere_plus:
                    rest_ap +=1
                if word in atmosphere_minus:
                    rest_am +=1
            df_restaurants.loc[i,'fp'] = rest_fp
            df_restaurants.loc[i,'fm'] = rest_fm
            df_restaurants.loc[i,'ssp'] = rest_ssp
            df_restaurants.loc[i,'ssm'] = rest_ssm
            df_restaurants.loc[i,'sqp'] = rest_sqp
            df_restaurants.loc[i,'sqm'] = rest_sqm
            df_restaurants.loc[i,'ap'] = rest_ap
            df_restaurants.loc[i,'am'] = rest_am
        except:
            pass
        print(f'Finished {counter} out of {df_restaurants.shape[0]}')
        counter += 1
    #dropping some restaurants that have NAs in their tokens
    for i in df_restaurants[df_restaurants['tokens'].isna()].index:
        df_restaurants.drop(i, inplace=True)

    for i in df_restaurants.columns:
        try:
            df_restaurants[i] = df_restaurants[i].astype(int)
        except:
            pass
    
    print('Calculating the dimension indexes')
    df_restaurants['food_quality_index'] = (df_restaurants['fp'] - df_restaurants['fm'])/(df_restaurants['fp'] + df_restaurants['fm'])
    df_restaurants['service_speed_index'] = (df_restaurants['ssp'] - df_restaurants['ssm'])/(df_restaurants['ssp'] + df_restaurants['ssm'])
    df_restaurants['service_quality_index'] = (df_restaurants['sqp'] - df_restaurants['sqm'])/(df_restaurants['sqp'] + df_restaurants['sqm'])
    df_restaurants['atmosphere_index'] = (df_restaurants['ap'] - df_restaurants['am'])/(df_restaurants['ap'] + df_restaurants['am'])

    df_restaurants['food_quality_index'] = 3 + df_restaurants['food_quality_index'] * 2 
    df_restaurants['service_speed_index'] = 3 + df_restaurants['service_speed_index'] * 2 
    df_restaurants['service_quality_index'] = 3 + df_restaurants['service_quality_index'] * 2 
    df_restaurants['atmosphere_index'] = 3 + df_restaurants['atmosphere_index'] * 2 
    
    #making the rounding to the nearest .5
    def rounding(num):
        whole = num // 1
        remainder = num - whole
        if remainder >= .75:
            remainder = 1
        elif remainder >= .5:
            remainder = .5
        elif remainder <= .25:
            remainder = 0
        else:
            remainder = .5
        return round((whole + remainder), 2)

    df_restaurants['food_quality_index'] = [rounding(i) for i in df_restaurants['food_quality_index']]
    df_restaurants['service_speed_index'] = [rounding(i) for i in df_restaurants['service_speed_index']]
    df_restaurants['service_quality_index'] = [rounding(i) for i in df_restaurants['service_quality_index']] 
    df_restaurants['atmosphere_index'] = [rounding(i) for i in df_restaurants['atmosphere_index']]
    
    df_restaurants.to_csv('./city_reviews_w_ratings.csv', index=False)



def model_bert_summary():
    #text summarization
    print('Loading the model')
    model = Summarizer()
    print('Loading datasets')
    df_bert = pd.read_csv('./final_reviews.csv')
    df_bus = pd.read_csv('./restaurants.csv')
    print('creating average reviews')
    counter = 1
    df_bert['avg_review'] = ['Not yet generated' for i in range(df_bert.shape[0])]
    for business in df_bus['business_id']:
        review_df = df_bert[df_bert['business_id']==business]
        total_review = ''
        for i in review_df['text']:
            total_review += (i + ' ')
            if len(total_review) > 999_600:
                break
        if review_df.shape[0] < 100:
            result = model(total_review, num_sentences=3, max_length=100)
        elif review_df.shape[0] > 100 and review_df.shape[0] < 1000:
            result = model(total_review, num_sentences=4, max_length=150)
        else:
            result = model(total_review, num_sentences=5, max_length=210)
        df_bus.loc[df_bus[df_bus['business_id'] == business].index, 'avg_review']=result
        print(f'Completed {counter} out of {df_bus.shape[0]}')
        counter += 1
    df_bus.to_csv('./city_average_reviews.csv', index=False)

    
if __name__ == '__main__':
    # PREPROCESSING THE DATA
    preprocessing()
    only_rest()
    more_details()
    combine()
    time = pd.to_datetime('01-01-2011')
    time_clean(time)
    rearrange()
    lang_clear()
    
    # Removing all the non-US reviews via multiprocessing
    print('Cleaning of non-US reviews')
    df = pd.read_csv('./eng_reviews.csv')
    states = pd.read_csv('./state_abbr.csv')
    states = list(states['usps'])
    print('Multipricessing Start')
    pool = mp.Pool(mp.cpu_count())
    pool.map(us_only, range(df.shape[0]))
    print('Saving final file')
    df.to_csv('./final_reviews.csv', index=False)
    print('Cleaning of non-US reviews COMPLETE')
    print('*************************')

    #Choosing a city as the whole dataset is too large
    city = 'Austin'
    state = 'TX'
    choose_city_state(city, state)

    #MODELLING
    model_token_lemm()
    dimensions()
    model_bert_summary()
