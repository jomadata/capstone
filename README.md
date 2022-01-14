
## DSI1011 General Assembly Data Science Immersive Capstone Project
### by Joomart Achekeev


#### Introduction

This is my Capstone project for my Data Science Immersive from General Assembly. Enjoy!

Initial idea of the project was to derrive additional rating dimensions from the reviews such as atmosphere and food quality to compliment simple five star rating system. During the process implementation an additional idea was to generate and average review, a mean() of all the text reviews.

Both goals were achieved with the use of a handmade 'sentiment analysis' typish approach to the text analysis and Huggingface's BERT model.https://huggingface.co/

#### Data

Yelp academic dataset (https://www.yelp.com/dataset) for 2019. Includes 6'835'403 (6.8 Gb) reviews of 160'585 (121 Mb) businesses. It includes such data as business ID, adress, reviews, reviewer id, review id, dates and other data.

This dataset is provided free of charge for academic purposes.

#### Preprocessing

Learned the hard way the general perception that most of the time Data Scientist are spending on data preparation, the data preprocessing was the main time consuming task as of this project.

Chunking: Even for a powerfull cloud computing instance the yelp dataset was too big. using chunking method, chunked up the dataset into 9 chunks with 1_000_000 reviews each.

Restaurants: Deleted all the reviews data that was not about the restaurants.

Combining: Combined all the processed chunks into single dataset.

Time: Data contained reviews dating back to 2004, to be on the safe side deleted every review that was before 01-01-2011.

Rearranging: Rearranged the columns and deleted the unneeded columns

Language: cleared non English reviews with langdetect library and its method .detect()

US only: Yelp data includes data for Canada also, created a filter and fed the whole dataset to it.

After all the modelling added a function that will filter out reviews for a certain city and state to improve the code implementation time
and demonstration of the results.

#### Modelling

The modelling includes 'handmade' dimensions analysis, similar to the sentiment analysis and BERT model fed with a single string from all reviews for a single business.

Dimensions
Dimension markers include synonims for such words as tasty, delicious, fast, slow, quick, comfortable and others from https://www.thesaurus.com/. These dimensions, after lemmatization are used to count the single string reviews for each business and are used to calculate through a series of transformations ratings on a 1 to 5 scale for food quality, service speed, service quality, and atmosphere (more thouroghly can be seen in the code)

BERT model. The special standalone summarize method of the model with certain hyperparameters is used to summarize the reviews into a single mean review. The review is then returned to the dataframe to its corresponding business_id.

#### Results

All the initial goals have been reached, even though the whole code was not fully executed as aproximately 40 days are needed to run the full code on the full Yelp dataset, even with the cloud computer with 32 cores and 128 Gb of RAM.

Additional dimensions such as food quality, service speed, service quality, and atmosphere have been derived from the separate selfcreated dataset. Moreover, with the use of the Huggungface's BERT model, a 'mean' review is derrived from the same reviews, with some interesting results.

Side plans such as Streamlit integration or use of GAN models was not implemented due to time limitation and huge amounts of data.

#### Examples
##### The Green Mesquite BBQ (4 star yelp rating)
Had a late lunch at 2 pm on a Tuesday with my husband. I would definitely visit them again if I am in the area and craving BBQ. I am not a fan of get your meat and add your sause, same with Bill Miller...What happened to BBQ on the Actual Pit? I didn't like the BBQ sauce that was on it.

Food Quality Index: 3.5
Service Speed Index: 3.5
Service Quality Index: 4.5
Atmosphere Index: 2.3

##### Habana Soco Restaurant (4 star yelp rating)
Growing up, my mom had a friend who was from Key West. I was thinking it was Mexican food, but when the chips and salsa didn't appear, I realized I was not in the typical "tex-mex" venue we often hit. I recommend to the ceviche and the Maduros Frito as appetizers as this the first place I've eaten in the US that tastes like home. The restaurant has cuban employees and atmosphere feels good but once our food arrives ropa vieja was salty ... way too much salty.

Food Quality Index: 4.5
Service Speed Index: 2
Service Quality Index: 4.5
Atmosphere Index: 3.5

##### Titaya's Thai Cuisine (5 star yelp rating)
Yummy, yummy, yummy - my favorite Thai restaurant in Austin! My favorite is the Gang Dang curry (with chicken but the tofu is amazing here) and the Jungle curry is nice, but my friend who LOVES spicy stuff said it wasn't as hot as he'd like. Pad see ew is always a must for me and I can say that this place did the dish justice but it wasn't over the top extraordinary. Either way, I know I will have a great meal in a great restaurant. If I lived here, this would be my go-to spot for Thai food.

Food Quality Index: 4.5
Service Speed Index: 3.5
Service Quality Index: 4.5
Atmosphere Index: 3.5


#### Lessons Learnt

1. Multiprocessing used in this project is slightly faster then the regular for loop. In depth understanding of the python implementation.
2. Big amounts of data are problematic even for cloud computing. Thus for personal projects use only managable data amounts.
3. In case there is no way to work without big data work in chunks.
4. Think about the way to give more weight to more recent reviews in comparison to the earlier ones.