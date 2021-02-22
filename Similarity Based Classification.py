#####################
#Faraz Moghimi, # Mahdi Behroozikhah
######################

# Importing libraries
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
#defining the pipe
clf = Pipeline([('scaler', StandardScaler()), ('clf', SVC(gamma='auto', verbose=True, kernel="rbf"))])
#clf = Pipeline([('clf', SVC(gamma='auto', verbose=True, kernel="rbf"))])
# clf = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
# clf = Pipeline([('clf', LogisticRegression())])

np.random.seed(seed=5) # fixing random parameters
#Importing the data
interactions = pd.read_csv(r'C:\Users\faraz\Desktop\Machine Learning course\Project\DB_Books.csv')  #
#defining train, test
interactions_train, interactions_validation_train = interactions[:190000], interactions[190000:]

# Creating a list of the books read by every individual
userBooks = defaultdict(list)  #create a dictionary with list data structure as a default for every key
for user,b,_ in interactions.values:
  userBooks[user].append(b)

#The goal in this segment is to create a dummy test set that consists of random users that books that they have not read.
book_list = interactions['bookID'].unique() #listing all the unique books
unread_list = []

#For each user and book pair the class is 1 right now. So, we wanna generate some random unread books with 0 class labels
#  getting random book, checking if that users has read the book, then maintaining it in the undread_list if the user has not read that book
for user,_,_ in interactions_validation_train.values:
  b = np.random.choice(book_list)
  while b in userBooks[user]:
    b = np.random.choice(book_list)
  unread_list.append([user, b, 0])
# the 10000 data set with 0 class labels are created

# fixing the type
interactions_validation_train = np.concatenate((interactions_validation_train.values, unread_list))
interactions_validation_train[:, 2] = interactions_validation_train[:, 2].astype(int)

#Basically, counting the popularity of a book: Seeing how many times it has been read
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in interactions_train.values:
  bookCount[book] += 1
  totalRead += 1

#creating a new array sorted by popularity based on book name.
mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

#Creating Ranking system that addresses the gaps between popularity:
#for example when the most popular book that has been read 500 times is ranked 1 the second most popular book would be ranked 501
count = 0
rank = defaultdict(int)
cnt = 1
for ic, i in mostPopular:
  count += ic
  rank[i] = cnt
  cnt+=ic




# Determining the book saviness of a user by counting the number of books that he has read
userCount = defaultdict(int)

#Counting the number of books read by a user
for user,book,_ in interactions_train.values:
  userCount[user] += 1
#Sorting the number of books read by a user
mostPopular = [(userCount[x], x) for x in userCount]
mostPopular.sort()
mostPopular.reverse()

#Ranking the number of books read by a user
count = 0
rank_u = defaultdict(int)
cnt = 1
for ic, i in mostPopular:
  count += ic
  rank_u[i] = cnt
  cnt+=ic



#Creating a set of books read for every user and users for every book
bookUsers = defaultdict(set)
usersbook_set = defaultdict(set)
for u, b, _ in interactions_train.values:
    bookUsers[b].add(u)
    usersbook_set[u].add(b)


#Creating a list of books read for each user in the training set
userBooks = defaultdict(list)
for user,b,_ in interactions_train.values:
  userBooks[user].append(b)

#Counting the number of times that a  user hast read 2 specific books for every two book selection --- Indicating Similarity between books
book_pair = defaultdict(dict)
user_pair = defaultdict(dict)

for u in bookUsers.keys():
  for b1 in bookUsers[u]:
    for b2 in bookUsers[u]:
      if b2 in user_pair[b1].keys():
        user_pair[b1][b2] += 1
      else:
        user_pair[b1][b2] = 1
#Counting the number of times that a book has been read by a pair of 2 particular users  ----- Indicating similarity between users
for u in userBooks.keys():
  for b1 in userBooks[u]:
    for b2 in userBooks[u]:
      if b2 in book_pair[b1].keys():
        book_pair[b1][b2] += 1
      else:
        book_pair[b1][b2] = 1

print("end of the long fors!") # finding out where the code running is at!!!

#Jaccard Similarity for books: Using Jaccard  between the the set of users that any 2 particular books have: Intersect/Union
def similarity(book, b):
  return len(bookUsers[book].intersection(bookUsers[b])) / len(bookUsers[book].union(bookUsers[b]))

#Jaccard Similarity for users: Using Jaccard algorithms between the the set of books read by any two users: Intersect/Union
def similarity_user(user, u):
  return len(usersbook_set[user].intersection(usersbook_set[u])) / len(usersbook_set[user].union(usersbook_set[u]))


# Extracting the 8 features that we have created so far in a classifiable manner
def parameters(u, b):
  mx_sim = 0
  p1, p = 0, 0
  #Creating the similarity feature: if mx_sim is high means the probability of the user reading b is higher
  for book in usersbook_set[u]:
    mx_sim = max(mx_sim, similarity(book, b))
    #Creating the probility feature: Gives us a popularity prospect in respect to the books the user have read
    if b in book_pair[book]:
      p += np.log(book_pair[book][b]/(len(bookUsers[b]))) #log(p(b, book) * p(b, book1) * ... *p[b, bookn)) for every book that user has read
  #Basically from the user prespective features down
  # Creating the similarity feature: if mx_sim1 is high means the probability of the user being similar
  # to other users who have read the book is high;hence, more likely that the user have read it
  mx_sim1 = 0
  for user in bookUsers[b]:
    mx_sim1 = max(mx_sim1, similarity_user(user, u))
    #Creating the probability feature: Gives us a popularity prospect in respect to the similarity of the particular users to other another user who have
    #who have read the book
    if user in user_pair[u]:
      p1 += np.log(user_pair[u][user]/(len(usersbook_set[user])))

  #Extracting the 8 features
  # return [(bookCount[b] / totalRead), (len(usersbook_set[u]) / totalRead),]
  # return [rank[b], rank_u[u], (bookCount[b] / totalRead), (len(usersbook_set[u]) / totalRead), ]
  # return [mx_sim, mx_sim1, rank[b], rank_u[u], (bookCount[b] / totalRead), (len(usersbook_set[u]) / totalRead)]
  return [mx_sim, mx_sim1, rank[b], rank_u[u],  (bookCount[b] / totalRead), (len(usersbook_set[u]) / totalRead), p, p1]

from sklearn.utils import shuffle
interactions_validation_train = shuffle(interactions_validation_train)
param = []
for u, b, _ in interactions_validation_train:
  param.append(parameters(u, b))

validation_set, test_set = interactions_validation_train[:15000], interactions_validation_train[15000:]
param_validation, param_test = param[:15000], param[15000:]
#Running the machine learning algorithm
clf.fit(param_validation, validation_set[:, 2].astype(bool))
print(accuracy_score(clf.predict(param_test), test_set[:, 2].astype(bool)))
