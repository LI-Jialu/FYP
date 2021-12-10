import datetime as datetime
import plotly.express as px
import numpy as np
import pickle
from sklearn.decomposition import PCA


from svm import svm_timepoint, svm_interval, train_test
from download_order_book import download_order_book as dob

class model_builder_2:
    def __init__(self, pcs, interval, path):
        self.pcs = pcs
        self.interval = interval
        self.path = path
        
    def load_model(self, file):
        try:
            fp=open(file,'rb')
            result=pickle.load(fp)
            fp.close()
        except Exception as e:
            result = 0
            print(e)
        return result
        
    def build_model(self):
        tt = train_test()
        DOB = dob()
        df_train = DOB.load_data(self.path, '2021-10-02')
        df_test = DOB.load_data(self.path, '2021-10-04')
        ##----------------------------------------SVM Single timepoint-------------------------------------------##
        starttime_point = datetime.datetime.now()
        #svm_s = svm_timepoint(df_train) # use the data of 2021-10-02 to train
        #f1, f2, f3, f4, f5, f6 = svm_s.timpoint_feature()
        #X_train = svm_s.generate_X(f1, f2, f3, f4, f5, f6)
        #y_train = svm_s.generate_y(f2)
        #svm_model_1 = tt.train(X_train, y_train)
        svm_model_1 = self.load_model('./models/model_1')
        
        svm_s = svm_timepoint(df_test) # use the data of 2021-10-04 to test
        f1, f2, f3, f4, f5, f6 = svm_s.timpoint_feature()
        X_test = svm_s.generate_X(f1, f2, f3, f4, f5, f6)
        y_test = svm_s.generate_y(f2)
        score_1, report_1 = tt.pred(svm_model_1, X_test, y_test)
        tt.dump(svm_model_1, 'model_1')
        endtime_point = datetime.datetime.now()
        print('The SVM using one-timestamp datapoint running time:')
        print((endtime_point - starttime_point).seconds)
        print('score:', score_1)
        print('report:')
        print(report_1)
        
        ##----------------------------------------SVM Interval (3 labels)---------------------------------------## 
        print('SVM Interval 3 label begins')
        starttime_interval = datetime.datetime.now()
        #svm_i_1 = svm_interval(df_train, self.interval) # use the data of 2021-10-02 to train
        #f1_1, f2_1, f3_1, f4_1, f5_1, f6_1, f7_1, f8_1 = svm_i_1.interval_feature()
        #X_train = svm_i_1.generate_X(f1_1, f2_1, f3_1, f4_1, f5_1, f6_1, f7_1, f8_1)
        #y_train = svm_i_1.generate_y(f2_1, f2_1.shape[0], 3)
        svm_model_2 = self.load_model('./models/model_2')
        
        svm_i_2 = svm_interval(df_test, self.interval) # use the data of 2021-10-04 to test
        f1_2, f2_2, f3_2, f4_2, f5_2, f6_2, f7_2, f8_2 = svm_i_2.interval_feature()
        X_test = svm_i_2.generate_X(f1_2, f2_2, f3_2, f4_2, f5_2, f6_2, f7_2, f8_2)
        y_test = svm_i_2.generate_y(f2_2, f2_2.shape[0], 3)
        score_2, report_2 = tt.pred(svm_model_2, X_test, y_test)
        tt.dump(svm_model_2, 'model_2')
        endtime_interval = datetime.datetime.now()
        print('The SVM using sliding window (3 labels) running time:')
        print((endtime_interval - starttime_interval).seconds)
        print('score:', score_2)
        print('report:')
        print(report_2)
        
        ##----------------------------------------SVM Interval (5 labels)--------------------------------------## 
        starttime_interval = datetime.datetime.now()
        #y_train = svm_i_1.generate_y(f2_1, f2_1.shape[0], 5)
        y_test = svm_i_2.generate_y(f2_2, f2_2.shape[0], 5)
        svm_model_3 = self.load_model('./models/model_3')
        score_3, report_3 = tt.pred(svm_model_3, X_test, y_test, 5)
        tt.dump(svm_model_3, 'model_3')
        endtime_interval = datetime.datetime.now()
        print('The SVM using sliding window (5 labels) running time:')
        print((endtime_interval - starttime_interval).seconds)
        print('score:', score_3)
        print('report:')
        print(report_3)
        
        ##-------------------------------------------SVM Interval with PCA-------------------------------------## 
        starttime_interval = datetime.datetime.now()
        pca = PCA(n_components = self.pcs)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        pca = PCA(n_components = self.pcs)
        pca.fit(X_test)
        X_test = pca.transform(X_test)
        svm_model_4 = self.load_model('./models/model_4')
        score_4, report_4 = tt.pred(svm_model_4, X_test, y_test, 5)
        tt.dump(svm_model_4, 'model_4')
        endtime_interval = datetime.datetime.now()
        print('The SVM using sliding window (5 labels) and PCA running time:')
        print((endtime_interval - starttime_interval).seconds)
        print('score:', score_4)
        print('report:')
        print(report_4)
        
        return [score_1, score_2, score_3, score_4,
                report_1, report_2, report_3, report_4]
        
class conditional_model_builder_2:
    def __init__(self, pcs, interval, path):
        self.pcs = pcs
        self.interval = interval
        self.path = path
        
    def load_model(self, file):
        try:
            fp=open(file,'rb')
            result=pickle.load(fp)
            fp.close()
        except Exception as e:
            result = 0
            print(e)
        return result
        
    def build_model(self):
        tt = train_test()
        DOB = dob()
        pcs = self.pcs
        interval = self.interval
        fng = [33, 27, 21, 47, 32, 42, 33, 50, 49, 43, 34, 54, 52, 71, 72, 
               74, 72, 74, 77, 75, 84, 75, 73, 71, 73, 73, 76, 73, 74, 74,
               73, 70, 66, 73, 76, 72, 73, 74, 75, 84, 82, 75, 78, 79, 78, 
               71, 70, 70, 78, 71, 71, 72, 74, 76, 68, 59, 54, 49, 54, 27, 
               20, 24, 25, 26, 27, 28, 33, 27, 21, 27, 50, 53, 50, 48, 53, 
               49, 30, 44, 32, 31, 46, 45, 47, 79, 79, 73, 72, 74, 74, 71, 
               73, 73, 72, 78, 71, 75, 73, 79, 79, 76, 78, 70, 70, 73, 72, 
               72, 71, 76, 70, 70, 70, 71, 65, 74, 69, 52, 50, 42, 48, 48, 
               60, 60, 53]
        fng.reverse()
        date_list = ['2021-07-30', '2021-07-31', '2021-08-01', '2021-08-02', '2021-08-03', 
                     '2021-08-04', '2021-08-05', '2021-08-06', '2021-08-07', '2021-08-08', 
                     '2021-08-09', '2021-08-10', '2021-08-11', '2021-08-12', '2021-08-13', 
                     '2021-08-14', '2021-08-15', '2021-08-16', '2021-08-17', '2021-08-18', 
                     '2021-08-19', '2021-08-20', '2021-08-21', '2021-08-22', '2021-08-23', 
                     '2021-08-24', '2021-08-25', '2021-08-26', '2021-08-27', '2021-08-28', 
                     '2021-08-29', '2021-08-30', '2021-08-31', '2021-09-01', '2021-09-02', 
                     '2021-09-03', '2021-09-04', '2021-09-05', '2021-09-06', '2021-09-07', 
                     '2021-09-08', '2021-09-09', '2021-09-10', '2021-09-11', '2021-09-12', 
                     '2021-09-13', '2021-09-14', '2021-09-15', '2021-09-16', '2021-09-17', 
                     '2021-09-18', '2021-09-19', '2021-09-20', '2021-09-21', '2021-09-22', 
                     '2021-09-23', '2021-09-24', '2021-09-25', '2021-09-26', '2021-09-27', 
                     '2021-09-28', '2021-09-29', '2021-09-30', '2021-10-01', '2021-10-02', 
                     '2021-10-03', '2021-10-04', '2021-10-05', '2021-10-06', '2021-10-07', 
                     '2021-10-08', '2021-10-09', '2021-10-10', '2021-10-11', '2021-10-12', 
                     '2021-10-13', '2021-10-14', '2021-10-15', '2021-10-16', '2021-10-17', 
                     '2021-10-18', '2021-10-19', '2021-10-20', '2021-10-21', '2021-10-22', 
                     '2021-10-23', '2021-10-24', '2021-10-25', '2021-10-26', '2021-10-27', 
                     '2021-10-28', '2021-10-29', '2021-10-30', '2021-10-31', '2021-11-01', 
                     '2021-11-02', '2021-11-03', '2021-11-04', '2021-11-05', '2021-11-06', 
                     '2021-11-07', '2021-11-08', '2021-11-09', '2021-11-10', '2021-11-11', 
                     '2021-11-12', '2021-11-13', '2021-11-14', '2021-11-15', '2021-11-16', 
                     '2021-11-17', '2021-11-18', '2021-11-19', '2021-11-20', '2021-11-21', 
                     '2021-11-22', '2021-11-23', '2021-11-24', '2021-11-25', '2021-11-26', 
                     '2021-11-27', '2021-11-28', '2021-11-29']
        
        X_train_greed = np.zeros((1,153), dtype = 'float64')
        y_train_greed = np.array([], dtype = 'float64')
        X_train_fear = np.zeros((1,153), dtype = 'float64')
        y_train_fear = np.array([], dtype = 'float64')
        starttime_interval = datetime.datetime.now()
        pca_fear = PCA(n_components = pcs)
        pca_greed = PCA(n_components = pcs)
        for i in range(len(date_list)):
            if(i == 62 or i == 95 or i == 7):
                continue
            df = DOB.load_data(self.path, date_list[i])
            svm_i = svm_interval(df, interval)
            f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
            X_train = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
            y_train = svm_i.generate_y(f2, f2.shape[0], 5)
            if(fng[i] < 50):
                X_train_fear = np.concatenate((X_train_fear, X_train))
                y_train_fear = np.append(y_train_fear, y_train)
            elif(fng[i] > 50):
                X_train_greed = np.concatenate((X_train_greed, X_train))
                y_train_greed = np.append(y_train_greed, y_train)
            
        X_train_greed = np.delete(X_train_greed, 0, axis = 0)
        X_train_fear = np.delete(X_train_fear, 0, axis = 0)
        
        pca_fear.fit(X_train_fear)
        pca_greed.fit(X_train_greed)
        tt.dump(pca_fear, 'pca_fear')
        exp_var_cumul = np.array(pca_fear.explained_variance_ratio_.cumsum())
        px.area(x=range(1, exp_var_cumul.shape[0] + 1),y=exp_var_cumul,labels={"x": "# Components", "y": "Explained Variance"})
        tt.dump(pca_greed, 'pca_greed')
        X_train_fear = pca_fear.transform(X_train_fear)
        X_train_greed = pca_greed.transform(X_train_greed)
        
        svm_model_fear = tt.train(X_train_fear, y_train_fear)
        tt.dump(svm_model_fear, 'model_fear')
        print('model_fear train done')
        svm_model_greed = tt.train(X_train_greed, y_train_greed)
        tt.dump(svm_model_greed, 'model_greed')
        print('model_greed train done')
        
        df = DOB.load_data(self.path, date_list[62])
        svm_i = svm_interval(df, interval)
        f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
        X_test_fear = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
        y_test_fear = svm_i.generate_y(f2, f2.shape[0], 5)
        X_test_fear = pca_fear.transform(X_test_fear)
        timestamp_fear = svm_i.get_timestamp_array()[:-1]
        
        df = DOB.load_data(self.path, date_list[95])
        svm_i = svm_interval(df, interval)
        f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
        X_test_greed = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
        y_test_greed = svm_i.generate_y(f2, f2.shape[0], 5)
        X_test_greed = pca_greed.transform(X_test_greed)
        timestamp_greed = svm_i.get_timestamp_array()[:-1]
        
        df = DOB.load_data(self.path, date_list[7])
        svm_i = svm_interval(df, interval)
        f1, f2, f3, f4, f5, f6, f7, f8 = svm_i.interval_feature()
        X_test_neutral = svm_i.generate_X(f1, f2, f3, f4, f5, f6, f7, f8)
        X_test_nf = pca_fear.transform(X_test_neutral)
        X_test_ng = pca_greed.transform(X_test_neutral)
        y_test_neutral = svm_i.generate_y(f2, f2.shape[0], 5)
        timestamp_neutral = svm_i.get_timestamp_array()[:-1]
        
        pred_y_greed, score_g, report_g = tt.pred(svm_model_greed, X_test_greed, y_test_greed, 5, True)
        pred_y_fear, score_f, report_f = tt.pred(svm_model_fear, X_test_fear, y_test_fear, 5, True)
        pred_y_neutral_fear, score_nf, report_nf = tt.pred(svm_model_fear, X_test_nf, y_test_neutral, 5, True)
        pred_y_neutral_greed, score_ng, report_ng = tt.pred(svm_model_greed, X_test_ng, y_test_neutral, 5, True)
        print(pred_y_greed)
        pred_y_greed = tt.attach_timestamp(pred_y_greed, timestamp_greed,'./metrics/train_g_test_g_5.csv')
        pred_y_fear = tt.attach_timestamp(pred_y_fear, timestamp_fear,'./metrics/train_f_test_f_5.csv')
        pred_y_neutral_fear = tt.attach_timestamp(pred_y_neutral_fear, timestamp_neutral,'./metrics/train_f_test_n_5.csv')
        pred_y_neutral_greed = tt.attach_timestamp(pred_y_neutral_greed, timestamp_neutral,'./metrics/train_g_test_n_5.csv')
        print(pred_y_greed)
        
        endtime_interval = datetime.datetime.now()
        print('The SVM using sliding window and PCA and Condition (Greed or Fear) running time:')
        print((endtime_interval - starttime_interval).seconds)
        print('score:', score_f, score_g, score_nf, score_ng)
        print('report:')
        print(report_f)
        print(report_g)
        print(report_nf)
        print(report_ng)
        
        return [pred_y_fear, pred_y_greed, pred_y_neutral_fear, pred_y_neutral_greed,
                score_f, score_g, score_nf, score_ng,
                report_f, report_g, report_nf, report_ng]
