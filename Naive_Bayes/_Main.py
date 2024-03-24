import math
# apply laplace smoothing method
class Naive_Bayes_CLF:
    def __init__(self, number_slable = 1, number_nslabel = 1):
        self.n_spam_label=number_slable
        self.n_non_spam_label=number_nslabel
        self.Spam_sum = []
        self.Non_spam_sum = []
        self.Total_spam_sum = None
        self.Total_non_spam_sum = None
        self.Features = []
        self.Spam_label_probability = float(self.n_spam_label / (self.n_spam_label + self.n_non_spam_label))
        self.Non_spam_label_probability = float(self.n_non_spam_label / (self.n_non_spam_label + self.n_spam_label))
        self.Cond_spam_prob = []
        self.Cond_non_spam_prob = []
        self.Complete_probability = {}
        
    # 0 = non_spam / 1 = spam
    def fit(self, X_train, y_train):
        self.Features = X_train.columns.tolist()
        X_train= X_train.values
        self.Total_spam_sum =0
        self.Total_non_spam_sum =0
        
        #tính tổng của từng features, và tổng của tổng từng features
        number_features =0
        for i in range(len(self.Features)):
            sum=0
            for y in range(self.n_non_spam_label):
                sum += X_train[y][i]
            self.Non_spam_sum.append(sum+1)
            self.Total_non_spam_sum += sum
            number_features +=1
            sum=0
            for z in range(self.n_non_spam_label,self.n_spam_label+self.n_non_spam_label):
                sum += X_train[z][i]
            self.Spam_sum.append(sum+1)
            self.Total_spam_sum += sum
        self.Total_non_spam_sum +=number_features
        self.Total_spam_sum +=number_features
        #p(x|c)
        self.Cond_spam_prob = [float((x+1) / self.Total_spam_sum) for x in self.Spam_sum]
        self.Cond_non_spam_prob = [float((x+1) / self.Total_non_spam_sum) for x in self.Non_spam_sum]
        combine_list = [[X,Y] for X,Y in zip(self.Cond_spam_prob,self.Cond_non_spam_prob)]
        self.Complete_probability = dict(zip(self.Features,combine_list))
        
                    
    def predict(self, X_test):
        p1 = math.log(self.Spam_label_probability)
        p2 = math.log(self.Non_spam_label_probability)
        for x in X_test:
            if x in self.Complete_probability:
                #print(str(self.Complete_probability[x][0])+ '+' +str(self.Complete_probability[x][1]) + '+')
                p1 += math.log(self.Complete_probability[x][0])
                p2 += math.log(self.Complete_probability[x][1])
            else:
                #print(str(float(1/self.Total_spam_sum)) + '+' + str(float(1/self.Total_non_spam_sum)) + '+')
                p1 += math.log(float(1/self.Total_spam_sum))
                p2 += math.log(float(1/self.Total_non_spam_sum))
        #print ('\n' + str(p2) + ' ' + str(p1))
        if p1 > p2 :
            return 1 
        else :
            return 0