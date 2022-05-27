import data
from gensim.models import word2vec

if __name__ == '__main__':
    # word2vec_model = word2vec.Word2Vec.load('./data/word2vec_model.model')
    # print('刘墉' in word2vec_model.wv.key_to_index)
    train_loader, val_loader, test_loader = data.online_shopping_10_cats()
    for data in train_loader:
        print(data)
        break