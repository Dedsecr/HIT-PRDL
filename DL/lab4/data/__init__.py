from data.jena_climate_2009_2016.jena_climate_2009_2016 import jena_climate_2009_2016
from data.online_shopping_10_cats.online_shopping_10_cats import online_shopping_10_cats
from data.process_word import process_word

def get_data(data):
    if data == 'jena_climate_2009_2016':
        return jena_climate_2009_2016
    if data == 'online_shopping_10_cats':
        return online_shopping_10_cats
    raise ValueError('Data {} not supported'.format(data))